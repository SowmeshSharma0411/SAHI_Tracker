import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from custom_ultralytics.custom_ultralytics.trackers.custom_byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.engine.results import Results
from queue import Queue
from threading import Thread
import concurrent.futures

# Load the YOLOv8 model
model = YOLO("best.pt")
model.to('cuda').half()  # Use half precision for faster inference

# Load the SAHI detection model without fusion
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="best.pt",
    confidence_threshold=0.1,
    device="cuda",
    load_at_init=False
)

# Manually load the model to avoid fusion
detection_model.model = model

# Load the tracker configuration
tracker_config_path = "bytetrack.yaml"
tracker_args = yaml_load(tracker_config_path)
tracker_args = IterableSimpleNamespace(**tracker_args)

# Open the video file
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\mot_ultralytics\\videoplayback (1).mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the BYTEtrack tracker
tracker = BYTETracker(tracker_args, frame_rate=fps)

# Measure the time taken for processing
start_time = time.time()

frame_count = 0
frame_skip = 2  # Process every 2nd frame
batch_size = 4  # Reduced batch size to avoid potential memory issues

# Create queues for frame processing pipeline
raw_frame_queue = Queue(maxsize=30)
detection_queue = Queue(maxsize=15)
tracking_queue = Queue(maxsize=15)

# Function to read frames from the video
def frame_reader():
    global frame_count
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
            batch_frames.append(frame)
        if batch_frames:
            raw_frame_queue.put(batch_frames)
        else:
            break
    raw_frame_queue.put(None)  # Signal end of frames

# Function to perform SAHI detection on a single frame
def sahi_detect(frame):
    try:
        sliced_results = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        boxes = []
        scores = []
        class_ids = []
        for object_prediction in sliced_results.object_prediction_list:
            bbox = object_prediction.bbox.to_xyxy()
            boxes.append(bbox)
            scores.append(object_prediction.score.value)
            class_ids.append(object_prediction.category.id)
        
        return frame, boxes, scores, class_ids
    except Exception as e:
        print(f"Error in SAHI detection: {e}")
        return frame, [], [], []

# Function to perform SAHI detection on a batch of frames
def sahi_detector():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            batch_frames = raw_frame_queue.get()
            if batch_frames is None:
                detection_queue.put(None)
                break
            
            # Process frames in parallel
            batch_results = list(executor.map(sahi_detect, batch_frames))
            detection_queue.put(batch_results)

# Function to perform tracking on a batch of detections
@torch.no_grad()
def tracker_processor():
    while True:
        batch_data = detection_queue.get()
        if batch_data is None:
            tracking_queue.put(None)
            break
        
        batch_results = []
        for data in batch_data:
            frame, boxes, scores, class_ids = data
            
            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float16, device='cuda')
                scores_tensor = torch.tensor(scores, dtype=torch.float16, device='cuda')
                class_ids_tensor = torch.tensor(class_ids, dtype=torch.int32, device='cuda')
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
                        results.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float16, device='cuda'))
            else:
                results = Results(orig_img=frame, path=video_path, names=model.names)
            
            batch_results.append(results)
        
        tracking_queue.put(batch_results)

# Start the pipeline threads
frame_reader_thread = Thread(target=frame_reader)
sahi_detector_thread = Thread(target=sahi_detector)
tracker_processor_thread = Thread(target=tracker_processor)

frame_reader_thread.start()
sahi_detector_thread.start()
tracker_processor_thread.start()

# Process the results
while True:
    batch_results = tracking_queue.get()
    if batch_results is None:
        break
    
    for results in batch_results:
        # Uncomment the following lines if you want to visualize the results
        plotted_frame = results.plot()
        cv2.imshow('Tracked Objects', plotted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        pass

# Wait for all threads to complete
frame_reader_thread.join()
sahi_detector_thread.join()
tracker_processor_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Measure and print the time taken for processing
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Frames processed: {frame_count}")
print(f"Average FPS: {frame_count / processing_time:.2f}")