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
import json

# Load the YOLOv8 model
model = YOLO("best.pt")
model.to('cuda').half()  # Use half precision for faster inference

# Load the SAHI detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="best.pt",
    confidence_threshold=0.05,
    device="cuda"
)

# Load the tracker configuration
tracker_config_path = "bytetrack.yaml"
tracker_args = yaml_load(tracker_config_path)
tracker_args = IterableSimpleNamespace(**tracker_args)

# Open the video file
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\mot_ultralytics\\videoplayback (1).mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(fps, total_frames)
# Initialize the BYTEtrack tracker with configuration
tracker = BYTETracker(tracker_args, frame_rate=fps)
start_time = time.time()
# Create queues for frame processing pipeline
raw_frame_queue = Queue(maxsize=120)
detection_queue = Queue(maxsize=80)
tracking_queue = Queue(maxsize=80)

# Global variable to store track history
track_history = {}

def normalize_bbox(x1, y1, x2, y2, img_width, img_height):
    return [
        max(0, min(x1 / img_width, 1)),
        max(0, min(y1 / img_height, 1)),
        max(0, min((x2 - x1) / img_width, 1)),
        max(0, min((y2 - y1) / img_height, 1))
    ]

def frame_reader():
    frame_count = 0
    frame_skip = 2  # Process every 3rd frame
    batch_size = 32
    while cap.isOpened():
        batch_frames = []
        for _ in range(batch_size):
            for _ in range(frame_skip):
                success, frame = cap.read()
                if not success:
                    break
            if not success:
                break
            frame_count += 1
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            batch_frames.append(resized_frame)
        if batch_frames:
            raw_frame_queue.put(batch_frames)
        else:
            break
    raw_frame_queue.put(None)  # Signal end of frames

def sahi_detector():
    while True:
        batch_frames = raw_frame_queue.get()
        if batch_frames is None:
            detection_queue.put(None)
            break
        
        batch_results = []
        for frame in batch_frames:
            sliced_results = get_sliced_prediction(
                image=frame,
                detection_model=detection_model,
                slice_height=240,
                slice_width=320,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25
            )
            
            boxes = []
            scores = []
            class_ids = []
            for object_prediction in sliced_results.object_prediction_list:
                bbox = object_prediction.bbox.to_xyxy()
                boxes.append(bbox)
                scores.append(object_prediction.score.value)
                class_ids.append(object_prediction.category.id)
            
            batch_results.append((frame, boxes, scores, class_ids))
        
        detection_queue.put(batch_results)

def tracker_processor():
    global track_history
    while True:
        batch_data = detection_queue.get()
        if batch_data is None:
            tracking_queue.put(None)
            break
        
        batch_results = []
        for data in batch_data:
            frame, boxes, scores, class_ids = data
            
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32).cuda(non_blocking=True)
            scores_tensor = torch.tensor(scores, dtype=torch.float32).cuda(non_blocking=True)
            class_ids_tensor = torch.tensor(class_ids, dtype=torch.int32).cuda(non_blocking=True)
            combined_data = torch.cat([boxes_tensor, scores_tensor.unsqueeze(1), class_ids_tensor.unsqueeze(1)], dim=1)
            
            results = Results(
                orig_img=frame,
                path=video_path,
                names=model.names,
                boxes=combined_data
            )
            
            det = results.boxes.cpu().numpy()
            if len(det):
                tracks = tracker.update(det, frame)
                if len(tracks):
                    idx = tracks[:, -1].astype(int)
                    results = results[idx]
                    results.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float32).cuda(non_blocking=True))
                    
                    # Extract and store tracking information
                    for track in tracks:
                        obj_id = int(track[-1])
                        x1, y1, x2, y2 = track[:4]
                        normalized_bbox = normalize_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
                        
                        if obj_id not in track_history:
                            track_history[obj_id] = []
                        track_history[obj_id].append(normalized_bbox)
            
            batch_results.append(results)
        
        tracking_queue.put(batch_results)

def save_track_history(output_file="track_history.json"):
    # Convert track_history to the desired format
    formatted_history = {str(obj_id): trajectory for obj_id, trajectory in track_history.items()}
    
    with open(output_file, 'w') as f:
        json.dump(formatted_history, f, indent=2)
    print(f"Track history saved to {output_file}")
    print(f"Total number of unique object IDs: {len(formatted_history)}")

# Start the pipeline threads
frame_reader_thread = Thread(target=frame_reader)
sahi_detector_thread = Thread(target=sahi_detector)
tracker_processor_thread = Thread(target=tracker_processor)

frame_reader_thread.start()
sahi_detector_thread.start()
tracker_processor_thread.start()

# Process the results
try:
    while True:
        batch_results = tracking_queue.get()
        if batch_results is None:
            break
        
        for results in batch_results:
            # Uncomment the following lines if you want to visualize the results
            plotted_frame = results.plot()
            cv2.imshow('Tracked Objects', plotted_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
            pass
except KeyboardInterrupt:
    print("Processing interrupted by user.")

# Wait for all threads to complete
frame_reader_thread.join()
sahi_detector_thread.join()
tracker_processor_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save track history to JSON file
save_track_history()
# Print processing statistics
print(f"Total number of unique object IDs tracked: {len(track_history)}")
# Print processing statistics
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Average FPS: {int(total_frames / processing_time)}")