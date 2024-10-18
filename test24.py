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
from scipy.spatial import cKDTree
import json

# Load the YOLOv8 model (using a lighter model)
model = YOLO("best.pt")
model.to('cuda').half()  # Use half precision for faster inference

# Load the SAHI detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="best.pt",
    confidence_threshold=0.15,
    device="cuda"
)

# Load the tracker configuration
tracker_config_path = "bytetrack.yaml"
tracker_args = yaml_load(tracker_config_path)
tracker_args = IterableSimpleNamespace(**tracker_args)

# Open the video file
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\mot_ultralytics\\videoplayback (1).mp4"
#video_path = "D:\\Sowmesh\\BACK GATE\\D24_20240325020738.mp4"
root_dir, file_name_with_ext = os.path.split(video_path)
file_name_to_save, ext = os.path.splitext(file_name_with_ext)

# video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\SampleVids\\bombay_trafficShortened.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(fps, total_frames)

# Initialize the BYTEtrack tracker with configuration
tracker = BYTETracker(tracker_args, frame_rate=fps)

# Measure the time taken for processing
start_time = time.time()

frame_count = 0
frame_skip = 2  # Increase frame skip to process every 3rd frame
batch_size = 32  # Increase batch size to process more frames at once

track_hist_dict = {}
# Create queues for frame processing pipeline
raw_frame_queue = Queue(maxsize=120)
detection_queue = Queue(maxsize=120)
tracking_queue = Queue(maxsize=120)

prev_positions = {}

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
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            batch_frames.append(resized_frame)
        if batch_frames:
            raw_frame_queue.put(batch_frames)
        else:
            break
    raw_frame_queue.put(None)  # Signal end of frames

# Function to perform SAHI detection on a batch of frames
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
            
            if not sliced_results.object_prediction_list:  # Check for no detections
                batch_results.append((frame, [], [], []))  # Append empty detections
                continue
            
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
    
    detection_queue.put(None)

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
            
            if not boxes:  # Check for no detections
                # You may want to handle this case differently if needed
                batch_results.append(Results(orig_img=frame, path=video_path, names=model.names, boxes=torch.empty((0, 6))))
                continue
            
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
                if len(tracks):
                    idx = tracks[:, -1].astype(int)
                    results = results[idx]
                    results.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float16).cuda(non_blocking=True))
            
            batch_results.append(results)
        
        tracking_queue.put(batch_results)
    
    tracking_queue.put(None)

# Function to display tracked frames
def display_frames():
    while True:
        batch_results = tracking_queue.get()
        if batch_results is None:
            break
    
        for results in batch_results:
            plotted_frame = results.plot()
            cv2.imshow('Tracked Objects', plotted_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Function to count nearby objects based on positions
def count_nearby_objects(current_position, all_objects, distance_threshold):
    if len(all_objects) < 2:
        return 0

    tree = cKDTree(all_objects)
    neighbors = tree.query_ball_point(current_position, distance_threshold)
    return len(neighbors) - 1

# Function to calculate velocity and movement
def calculate_velocity(current_position, obj_id):
    if obj_id in prev_positions:
        prev_x, prev_y = prev_positions[obj_id]
        dx = abs(current_position[0] - prev_x)
        dy = abs(current_position[1] - prev_y)
        velocity = np.sqrt(dx**2 + dy**2)
    else:
        dx, dy, velocity = 0, 0, 0
    
    prev_positions[obj_id] = current_position
    return dx, dy, velocity

# Function to update tracking history
def track_hist_func():
    global track_hist_dict
    global prev_positions
    
    all_positions = []
    distance_threshold = 50
    
    while True:
        batch_results = tracking_queue.get()
        if batch_results is None:
            break
        
        for results in batch_results:
            
            if results is None or results.boxes is None or len(results.boxes) == 0:
                continue
            
            if hasattr(results.boxes, 'id') and hasattr(results.boxes, 'xywhn'):
                all_positions = []
                if results.boxes.id is not None:
                    ids_list = results.boxes.id.tolist()
                else:
                    continue
                # ids_list = results.boxes.id.tolist()
                current_positions = results.boxes.xywhn[:, :4].tolist()
                
                for i, obj_id in enumerate(ids_list):
                    current_position = current_positions[i][:2]
                    dx, dy, velocity = calculate_velocity(current_position, obj_id)
                    all_positions.append(current_position)
                    nearby_count = count_nearby_objects(current_position, all_positions, distance_threshold)
                    
                    if obj_id in track_hist_dict.keys():
                        track_hist_dict[obj_id].append([
                            current_positions[i],
                            dx, dy, velocity, nearby_count
                        ])
                    else:
                        track_hist_dict[obj_id] = [[
                            current_positions[i],
                            dx, dy, velocity, nearby_count
                        ]]

# Start the pipeline threads
frame_reader_thread = Thread(target=frame_reader)
sahi_detector_thread = Thread(target=sahi_detector)
tracker_processor_thread = Thread(target=tracker_processor)
display_thread = Thread(target=display_frames)
track_hist_thread = Thread(target=track_hist_func)

frame_reader_thread.start()
sahi_detector_thread.start()
tracker_processor_thread.start()
display_thread.start()
track_hist_thread.start()

# Wait for all threads to complete
frame_reader_thread.join()
sahi_detector_thread.join()
tracker_processor_thread.join()
display_thread.join()
track_hist_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the tracking history to a JSON file
json_save_dir = "D:\\Sowmesh\\jsons_save"
output_file_path = json_save_dir + "\\" + file_name_to_save + ".json"
with open(output_file_path, 'w') as json_file:
    json.dump(track_hist_dict, json_file, indent=4)

# Measure and print the time taken for processing
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Average FPS: {int(total_frames / processing_time)}")
print(f"Track history saved to {output_file_path}")
