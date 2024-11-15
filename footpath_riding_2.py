import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from custom_ultralytics.custom_ultralytics.trackers.custom_byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.engine.results import Results
from pathlib import Path
import os
import concurrent.futures
from queue import Queue
from threading import Thread

class FootpathDetectionSystem:
    def __init__(self, model_path, footpath_model_path, video_path, tracker_config_path="bytetrack.yaml"):
        """
        Initialize the Footpath Detection System
        
        Args:
            model_path (str): Path to the YOLOv8 model
            footpath_model_path (str): Path to the footpath segmentation model
            video_path (str): Path to the input video
            tracker_config_path (str): Path to the tracker configuration file
        """
        self.setup_models(model_path, footpath_model_path)
        self.setup_video(video_path)
        self.setup_tracker(tracker_config_path)
        self.initialize_queues()
        self.mask = None
        
    def setup_models(self, model_path, footpath_model_path):
        """Setup YOLOv8 and footpath models"""
        self.model = YOLO(model_path)
        self.model.to('cuda').half()
        self.model_footpath = YOLO(footpath_model_path)
        
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.05,
            device="cuda"
        )
        
    def setup_video(self, video_path):
        """Setup video capture and get video properties"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties - FPS: {self.fps}, Total Frames: {self.total_frames}")
        
    def setup_tracker(self, tracker_config_path):
        """Setup BYTETracker"""
        tracker_args = yaml_load(tracker_config_path)
        tracker_args = IterableSimpleNamespace(**tracker_args)
        self.tracker = BYTETracker(tracker_args, frame_rate=self.fps)
        
    def initialize_queues(self):
        """Initialize processing queues"""
        self.raw_frame_queue = Queue(maxsize=120)
        self.detection_queue = Queue(maxsize=80)
        self.tracking_queue = Queue(maxsize=80)
        
    def generate_intersection_mask(self, max_frames=10):
        """
        Generate intersection mask from multiple frames
        
        Args:
            max_frames (int): Maximum number of frames to process for mask generation
        """
        intersection_mask = None
        frame_count = 0
        
        while self.cap.isOpened() and frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            results = self.model_footpath(frame)
            masks = results[0].masks
            
            if masks is not None:
                for mask in masks.data:
                    mask_resized = cv2.resize(
                        mask.cpu().numpy(),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    binary_mask = (mask_resized > 0).astype(np.uint8) * 255
                    
                    if intersection_mask is None:
                        intersection_mask = binary_mask.copy()
                    else:
                        intersection_mask = cv2.bitwise_and(intersection_mask, binary_mask)
                        
            frame_count += 1
            
        if intersection_mask is not None:
            self.mask = cv2.resize(intersection_mask, (640, 360), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite("intersection_mask_resized.png", self.mask)
            print("Saved: intersection_mask_resized.png")
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        
    def check_bbox_in_mask(self, box, img_width=640, img_height=360):
        """
        Check if the center of a bounding box falls within the masked region
        
        Args:
            box (tensor): Normalized box coordinates [x_center, y_center, width, height]
            img_width (int): Width of the image
            img_height (int): Height of the image
            
        Returns:
            bool: True if center point falls in masked region, False otherwise
        """
        try:
            x_center, y_center = box[:2].tolist()
            x_pixel = int(x_center * img_width)
            y_pixel = int(y_center * img_height)
            
            x_pixel = max(0, min(x_pixel, img_width - 1))
            y_pixel = max(0, min(y_pixel, img_height - 1))
            
            if self.mask is None or self.mask.shape[:2] != (img_height, img_width):
                print(f"Warning: Mask dimensions {self.mask.shape if self.mask is not None else None} "
                      f"don't match expected dimensions ({img_height}, {img_width})")
                return False
                
            return self.mask[y_pixel, x_pixel] == 255
            
        except Exception as e:
            print(f"Error checking bbox in mask: {str(e)}")
            return False
            
    def frame_reader(self, frame_skip=2, batch_size=32):
        """Read frames from video in batches"""
        while self.cap.isOpened():
            batch_frames = []
            for _ in range(batch_size):
                for _ in range(frame_skip):
                    success, frame = self.cap.read()
                    if not success:
                        break
                if not success:
                    break
                resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                batch_frames.append(resized_frame)
            if batch_frames:
                self.raw_frame_queue.put(batch_frames)
            else:
                break
        self.raw_frame_queue.put(None)
        
    def sahi_detector(self):
        """Perform SAHI detection on batches of frames"""
        while True:
            batch_frames = self.raw_frame_queue.get()
            if batch_frames is None:
                self.detection_queue.put(None)
                break
            
            batch_results = []
            for frame in batch_frames:
                sliced_results = get_sliced_prediction(
                    image=frame,
                    detection_model=self.detection_model,
                    slice_height=240,
                    slice_width=320,
                    overlap_height_ratio=0.25,
                    overlap_width_ratio=0.25
                )
                
                boxes = []
                scores = []
                class_ids = []
                for prediction in sliced_results.object_prediction_list:
                    boxes.append(prediction.bbox.to_xyxy())
                    scores.append(prediction.score.value)
                    class_ids.append(prediction.category.id)
                
                batch_results.append((frame, boxes, scores, class_ids))
            
            self.detection_queue.put(batch_results)
            
    @torch.no_grad()
    def tracker_processor(self):
        """Process tracking on batches of detections"""
        while True:
            batch_data = self.detection_queue.get()
            if batch_data is None:
                self.tracking_queue.put(None)
                break
            
            batch_results = []
            for data in batch_data:
                frame, boxes, scores, class_ids = data
                
                boxes_tensor = torch.tensor(boxes, dtype=torch.float16).cuda(non_blocking=True)
                scores_tensor = torch.tensor(scores, dtype=torch.float16).cuda(non_blocking=True)
                class_ids_tensor = torch.tensor(class_ids, dtype=torch.int32).cuda(non_blocking=True)
                combined_data = torch.cat([boxes_tensor, scores_tensor.unsqueeze(1), 
                                         class_ids_tensor.unsqueeze(1)], dim=1)
                
                results = Results(
                    orig_img=frame,
                    path=self.video_path,
                    names=self.model.names,
                    boxes=combined_data
                )
                
                det = results.boxes.cpu().numpy()
                if len(det):
                    tracks = self.tracker.update(det, frame)
                    if len(tracks):
                        idx = tracks[:, -1].astype(int)
                        results = results[idx]
                        results.update(boxes=torch.as_tensor(tracks[:, :-1], 
                                                          dtype=torch.float16).cuda(non_blocking=True))
                
                batch_results.append(results)
            
            self.tracking_queue.put(batch_results)
            
    def process_video(self):
        """Main processing loop"""
        print("Starting video processing...")
        start_time = time.time()
    
        # Start processing threads
        frame_reader_thread = Thread(target=self.frame_reader)
        sahi_detector_thread = Thread(target=self.sahi_detector)
        tracker_processor_thread = Thread(target=self.tracker_processor)
    
        frame_reader_thread.start()
        sahi_detector_thread.start()
        tracker_processor_thread.start()
    
    # Process results
        while True:
            batch_results = self.tracking_queue.get()
            if batch_results is None:
                break
        
            for results in batch_results:
                for result in results:
                # Get boxes data
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                
                # Get normalized coordinates and class indices
                    xywhn = boxes.xywhn
                    cls = boxes.cls.int()  # Get class indices
                
                    for i, box in enumerate(xywhn):
                        if self.check_bbox_in_mask(box):
                            class_idx = int(cls[i])
                            result.names[class_idx] = 'footpath riding'
            
                plotted_frame = results.plot()
                cv2.imshow('Tracked Objects', plotted_frame)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Wait for threads to complete
        frame_reader_thread.join()
        sahi_detector_thread.join()
        tracker_processor_thread.join()
    
    # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
    
    # Print processing statistics
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {int(self.total_frames / processing_time)}")

def main():
    # Configuration
    MODEL_PATH = "best.pt"
    FOOTPATH_MODEL_PATH = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\footpath_seg_framework\\footpath_seg_trained\\train4\\weights\\footpath_best.pt"
    VIDEO_PATH = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\footpath_seg_framework\\vlc-record-2024-09-23-10h50m15s-D05_20240325070810.mp4-.mp4"
    
    # Initialize system
    system = FootpathDetectionSystem(
        model_path=MODEL_PATH,
        footpath_model_path=FOOTPATH_MODEL_PATH,
        video_path=VIDEO_PATH
    )
    
    # Generate intersection mask
    print("Generating intersection mask...")
    system.generate_intersection_mask()
    
    # Process video
    system.process_video()

if __name__ == "__main__":
    main()