import cv2
import torch
import time
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.engine.results import Results
from pathlib import Path

# Start timing
start_time = time.time()

# Load the YOLOv8 model
model = YOLO("C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\models\\runs\\detect\\train15\\weights\\best.pt")

# Load the SAHI detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\models\\runs\\detect\\train15\\weights\\best.pt",
    confidence_threshold=0.05,
    device="cuda"
)

# Load the tracker configuration
tracker_config_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\bytetrack.yaml"
tracker_args = yaml_load(tracker_config_path)
tracker_args = IterableSimpleNamespace(**tracker_args)

# Open the video file
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\SampleVids\\traffic_vid2Shortened.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the BYTEtrack tracker with configuration
tracker = BYTETracker(tracker_args, frame_rate=fps)

# Initialize video writer
output_path = "output_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Process every 5th frame
    if frame_count % 2 == 0:
        # Step 1: Use SAHI to get detections on sliced images
        sliced_results = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.75,
            overlap_width_ratio=0.75
        )
        
        boxes = []
        scores = []
        class_ids = []
        for object_prediction in sliced_results.object_prediction_list:
            bbox = object_prediction.bbox.to_xyxy()
            boxes.append(bbox)
            scores.append(object_prediction.score.value)
            class_ids.append(object_prediction.category.id)
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        class_ids_tensor = torch.tensor(class_ids, dtype=torch.int64)
        combined_data = torch.cat([boxes_tensor, scores_tensor.unsqueeze(1), class_ids_tensor.unsqueeze(1)], dim=1)
        
        results = Results(
            orig_img=frame,
            path=video_path,
            names=model.names,
            boxes=combined_data
        )
        
        # Step 2: Update tracker with detections
        det = results.boxes.cpu().numpy()
        if len(det):
            tracks = tracker.update(det, frame)
            
            # Update results with tracking information
            if len(tracks):
                idx = tracks[:, -1].astype(int)
                results = results[idx]
                results.update(boxes=torch.as_tensor(tracks[:, :-1]))
        
        # Step 3: Plot results
        plotted_frame = results.plot()
    else:
        # For frames we're skipping, just use the original frame
        plotted_frame = frame
    
    # Write the frame to the output video
    out.write(plotted_frame)
    
    # Display the frame (optional)
    cv2.imshow('Tracked Objects', plotted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate and print the total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Tracking completed. Output saved to {output_path}")
print(f"Total execution time: {total_time:.2f} seconds")