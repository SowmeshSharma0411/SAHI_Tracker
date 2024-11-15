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

model_footpath = YOLO("C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\footpath_seg_framework\\footpath_seg_trained\\train4\\weights\\footpath_best.pt")
# Video path
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\footpath_seg_framework\\vlc-record-2024-09-23-10h50m15s-D05_20240325070810.mp4-.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
max_frames = 10

# Initialize a mask for intersection
intersection_mask = None

# Process the video frames to generate the intersection mask
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference on the frame
    results = model_footpath(frame)
    
    # Get the masks from the results
    masks = results[0].masks

    if masks is not None:
        # Loop over all masks (objects detected)
        for mask in masks.data:
            # Resize the mask to match the frame size
            mask_resized = cv2.resize(mask.cpu().numpy(), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Convert the mask to a binary mask: values greater than 0 are set to 255 (white), others set to 0 (black)
            binary_mask = (mask_resized > 0).astype(np.uint8) * 255
            
            # Save the binary mask (it will only have the detected object in white, and the rest will be black)
            mask_filename = f"binary_mask_frame_{frame_count + 1}.png"
            cv2.imwrite(mask_filename, binary_mask)
            print(f"Saved: {mask_filename}")
            
            # Update intersection mask (if required)
            if intersection_mask is None:
                intersection_mask = binary_mask.copy()
            else:
                intersection_mask = cv2.bitwise_and(intersection_mask, binary_mask)

    frame_count += 1

# After processing all frames, save the intersection mask (if it exists)
if intersection_mask is not None:
    # Resize the intersection mask to 640x360
    resized_intersection_mask = cv2.resize(intersection_mask, (640, 360), interpolation=cv2.INTER_NEAREST)
    
    # Save the resized intersection mask
    intersection_filename = "intersection_mask_resized.png"
    cv2.imwrite(intersection_filename, resized_intersection_mask)
    print(f"Saved: {intersection_filename}")

# Release the video capture object
cap.release()

# Load the intersection mask
mask_path = "intersection_mask_resized.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

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
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total Frames: {total_frames}")

# Initialize the BYTEtrack tracker with configuration
tracker = BYTETracker(tracker_args, frame_rate=fps)

# Measure the time taken for processing
start_time = time.time()

frame_count = 0
frame_skip = 2  # Increase frame skip to process every 3rd frame
batch_size = 32  # Process 32 frames in a batch

# Create queues for frame processing pipeline
raw_frame_queue = Queue(maxsize=120)  # Increase queue size
detection_queue = Queue(maxsize=80)
tracking_queue = Queue(maxsize=80)

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
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)  # Resize to 640x360
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
                slice_height=240,  # Reduce slice size
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
            if len(det):
                tracks = tracker.update(det, frame)
                if len(tracks):
                    idx = tracks[:, -1].astype(int)
                    results = results[idx]
                    results.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float16).cuda(non_blocking=True))
            
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
        for result in results:
            # Get bounding box details: xywhn are normalized (0-1 range)
            xywhn = result.boxes.xywhn  # xywhn = [x_center, y_center, width, height] (normalized)
        
            for box in xywhn:
                # Convert normalized coordinates to pixel values (assuming image size is 640x360)
                x_center, y_center, width, height = box.tolist()
                x_center = int(x_center * 640)  # Rescale to 640px width
                y_center = int(y_center * 360)  # Rescale to 360px height

                # Check if the center of the bounding box lies in the white region of the intersection_mask
                if mask[y_center, x_center] == 255:  # White region in mask
                    # If inside the white region, change the class name to 'footpath riding'
                    result.names[result.cls] = 'footpath riding'  # Assuming 'cls' refers to class ID
                    print(f"Object at ({x_center}, {y_center}) is in the white region. Class changed to 'footpath riding'.")
        
        # Plot the updated frame (with modified class names)
        plotted_frame = results.plot()

        # Show the frame with updated class names
        cv2.imshow('Tracked Objects', plotted_frame)

        # Check for 'q' key press to break the loop
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
print(f"Average FPS:", int(total_frames / processing_time))
