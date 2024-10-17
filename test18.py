import cv2
import torch
import time
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from custom_ultralytics.custom_ultralytics.trackers.custom_byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.engine.results import Results
from queue import Queue
from threading import Thread

# Load a smaller YOLO model
model = YOLO("best.pt").to('cuda').half()

# Load the SAHI detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="best.pt",
    confidence_threshold=0.1,
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
fps = int(cap.get(cv2.CAP_PROP_FPS))
tracker = BYTETracker(tracker_args, frame_rate=fps)

# Processing variables
raw_frame_queue = Queue(maxsize=120)
detection_queue = Queue(maxsize=80)
tracking_queue = Queue(maxsize=80)

def frame_reader():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        resized_frame = cv2.resize(frame,  (1037, 583), interpolation=cv2.INTER_AREA)  # Reduced size
        raw_frame_queue.put(resized_frame)
    raw_frame_queue.put(None)

def sahi_detector():
    while True:
        frame = raw_frame_queue.get()
        if frame is None:
            detection_queue.put(None)
            break

        # Process every 2nd frame
        if (raw_frame_queue.qsize() % 2) == 0:
            sliced_results = get_sliced_prediction(
                image=frame,
                detection_model=detection_model,
                slice_height=315,
                slice_width=560,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25
            )
            
            boxes, scores, class_ids = [], [], []
            for obj in sliced_results.object_prediction_list:
                boxes.append(obj.bbox.to_xyxy())
                scores.append(obj.score.value)
                class_ids.append(obj.category.id)
            
            detection_queue.put((frame, boxes, scores, class_ids))

@torch.no_grad()
def tracker_processor():
    while True:
        data = detection_queue.get()
        if data is None:
            tracking_queue.put(None)
            break
        
        frame, boxes, scores, class_ids = data
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float16).cuda(non_blocking=True)
            scores_tensor = torch.tensor(scores, dtype=torch.float16).cuda(non_blocking=True)
            class_ids_tensor = torch.tensor(class_ids, dtype=torch.int32).cuda(non_blocking=True)
            combined_data = torch.cat([boxes_tensor, scores_tensor.unsqueeze(1), class_ids_tensor.unsqueeze(1)], dim=1)
            
            # Pass the additional parameters
            results = Results(orig_img=frame, path=video_path, names=model.names, boxes=combined_data)
            det = results.boxes.cpu().numpy()
            if len(det):
                tracks = tracker.update(det, frame)
                if len(tracks):
                    idx = tracks[:, -1].astype(int)
                    results.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float16).cuda(non_blocking=True))
        
        tracking_queue.put(results)

# Start time measurement
start_time = time.time()

# Start threads
threads = [
    Thread(target=frame_reader),
    Thread(target=sahi_detector),
    Thread(target=tracker_processor)
]
for thread in threads:
    thread.start()

# Process results
while True:
    results = tracking_queue.get()
    if results is None:
        break
    plotted_frame = results.plot()  # Visualization
    cv2.imshow('Tracked Objects', plotted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for thread in threads:
    thread.join()

cap.release()
cv2.destroyAllWindows()

# Measure and print the total processing time
end_time = time.time()
total_time = end_time - start_time
print(f"Total processing time: {total_time:.2f} seconds")
