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
import json
from queue import Queue
from threading import Thread
from filterpy.kalman import KalmanFilter

# Load the YOLOv8 model (using a lighter model)
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
print(f"FPS: {fps}, Total Frames: {total_frames}")

# Initialize the BYTEtrack tracker with configuration
tracker = BYTETracker(tracker_args, frame_rate=fps)

# Kalman Filter initialization
def init_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 10
    kf.P *= 1000
    return kf

# Kalman update function
def update_kalman(kf, x, y):
    kf.predict()
    kf.update(np.array([x, y]))
    return kf.x

# Start the pipeline threads
frame_reader_thread = Thread(target=lambda: None)  # Placeholder
sahi_detector_thread = Thread(target=lambda: None)  # Placeholder
tracker_processor_thread = Thread(target=lambda: None)  # Placeholder

# Measure the time taken for processing
start_time = time.time()

frame_count = 0
frame_skip = 2  # Increase frame skip to process every 3rd frame
batch_size = 32

# Create queues for frame processing pipeline
raw_frame_queue = Queue(maxsize=120)
detection_queue = Queue(maxsize=80)
tracking_queue = Queue(maxsize=80)

# Store track history and filters
track_history = {}
kalman_filters = {}
dynamic_radius = 150
alpha = 0.2

# Frame reader function
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
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            batch_frames.append(resized_frame)
        if batch_frames:
            raw_frame_queue.put(batch_frames)
        else:
            break
    raw_frame_queue.put(None)  # Signal end of frames

# SAHI detector function
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

# Tracker processor function
@torch.no_grad()
def tracker_processor():
    global dynamic_radius
    frame_diagonal = np.sqrt(width**2 + height**2)

    while True:
        batch_data = detection_queue.get()
        if batch_data is None:
            tracking_queue.put(None)
            break
        
        batch_results = []
        for data in batch_data:
            frame, boxes, scores, class_ids = data
            
            boxes_tensor = torch.tensor(boxes, dtype=torch.float16).cuda(non_blocking=True)
            scores_tensor = torch.tensor(scores, dtype=torch.float16).cuda(non_blocking=True)
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
                for track in tracks:
                    track_id = int(track[-1])
                    x, y, w, h = track[:4]

                    # Initialize Kalman filter for new track IDs
                    if track_id not in kalman_filters:
                        kalman_filters[track_id] = init_kalman_filter()
                        track_history[track_id] = []

                    # Update Kalman filter and store history
                    kalman_state = update_kalman(kalman_filters[track_id], x, y)
                    smoothed_x, smoothed_y = kalman_state[0], kalman_state[1]
                    
                    # Calculate delta and speed
                    if len(track_history[track_id]) > 0:
                        last_x, last_y, _, _, _ = track_history[track_id][-1]
                        delta_x = smoothed_x - last_x
                        delta_y = smoothed_y - last_y
                        speed = np.sqrt(delta_x**2 + delta_y**2)
                    else:
                        delta_x, delta_y, speed = 0, 0, 0

                    # Count nearby objects
                    curr_frame_objects = [(smoothed_x, smoothed_y, w, h) for t in track_history.values() for t in t]
                    nearby_count = count_nearby_objects((smoothed_x, smoothed_y), curr_frame_objects, dynamic_radius)

                    # Store trajectory
                    track_history[track_id].append((float(smoothed_x), float(smoothed_y), float(w), float(h), delta_x, delta_y, speed, nearby_count))

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
        plotted_frame = results.plot()
        cv2.imshow('Tracked Objects', plotted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
print(f"Average FPS: {int(total_frames / processing_time)}")

# Save track history
output_path = "track_history.json"
with open(output_path, "w") as f:
    json.dump(track_history, f, indent=4)

print(f"Track history saved to {output_path}")
