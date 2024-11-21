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
# video_path = "D:\\Sowmesh\\yt_vids_2\\vids\\video_252.mp4"
video_path = r"D:\Sowmesh\MulitpleObjectTracking\SampleVids\bombay_trafficShortened.mp4"
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
#batch_size = 32  # Increase batch size to process more frames at once
batch_size = 32
# Create queues for frame processing pipeline
raw_frame_queue = Queue(maxsize=120)  # Increase queue size
detection_queue = Queue(maxsize=80)
tracking_queue = Queue(maxsize=80)

track_hist = {}
last_time = {}

# Queue to store expired objects
expired_objects_queue = Queue(maxsize=100)

log_file = "hit_and_run_log.txt"

def log_hit_and_run(message):
    """Log hit-and-run-related messages to a text file."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
   
    print(message)

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
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) #(640, 360) - 34 (1037, 583) - 23 (960, 540) - 29
            # # Reduce resolution
            #resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
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
                slice_height=240,  # Reduce slice size  #288 #240
                slice_width=320,    #512 #320
                overlap_height_ratio=0.25,  # Reduce overlap
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


def update_tracking_history(tracked_objects):
    global track_hist, last_time
    current_time = time.time()
    for obj_id, obj_data in tracked_objects.items():
        if obj_id in track_hist:
            track_hist[obj_id].append(obj_data)
        else:
            track_hist[obj_id] = [obj_data]
        last_time[obj_id] = current_time
        
def manage_expired_objects():
    while True:
        current_time = time.time()
        expired_ids = []
        for obj_id, last_detected in list(last_time.items()):
            if current_time - last_detected > 10:  # 10 seconds threshold
                expired_ids.append(obj_id)

        for obj_id in expired_ids:
            # Add to expired queue and remove from dictionaries
            expired_objects_queue.put((obj_id, track_hist[obj_id]))
            log_hit_and_run(f"Expired object: {obj_id}")
            
            del track_hist[obj_id]
            del last_time[obj_id]

        time.sleep(1)  # Check every second to avoid busy looping
        
expired_objects_manager_thread = Thread(target=manage_expired_objects, daemon=True)
expired_objects_manager_thread.start()


# Function to perform tracking on a batch of detections
@torch.no_grad()  # Disable gradient computation for inference
def tracker_processor():
    while True:
        batch_data = detection_queue.get()
        if batch_data is None:
            tracking_queue.put(None)
            break
        
        batch_results = []
        for data in batch_data:
            frame, boxes, scores, class_ids = data
            
            boxes_tensor = torch.tensor(boxes, dtype=torch.float16).cuda(non_blocking=True)  # Use float16
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
            tracked_objects = {}
            if len(det):
                tracks = tracker.update(det, frame)
                if len(tracks):
                    idx = tracks[:, -1].astype(int)
                    results = results[idx]
                    results.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float16).cuda(non_blocking=True))
                    # Update tracking history
                    for track in tracks:
                        obj_id = int(track[-1])
                        tracked_objects[obj_id] = track[:-1]
                        
            update_tracking_history(tracked_objects)  # Update the history in parallel
            
            batch_results.append(results)
        
        tracking_queue.put(batch_results)
        
def cleanup_remaining_objects():
    global track_hist, last_time
    current_time = time.time()
    for obj_id in list(track_hist.keys()):
        # Add all remaining tracked objects to the expired queue
        expired_objects_queue.put((obj_id, track_hist[obj_id]))
    
    # Clear the dictionaries
    track_hist.clear()
    last_time.clear()


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
        #pass  # Remove this line if you uncomment the visualization code

# Wait for all threads to complete
frame_reader_thread.join()
sahi_detector_thread.join()
tracker_processor_thread.join()
expired_objects_manager_thread.join()

cleanup_remaining_objects()

while not expired_objects_queue.empty():
    expired_object = expired_objects_queue.get()
    log_hit_and_run(f"Expired object: {expired_object}")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Measure and print the time taken for processing
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.2f} seconds")
#print(f"Frames processed: {frame_count}")
print(f"Average FPS:", int(total_frames / processing_time))