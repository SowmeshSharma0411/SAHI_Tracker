from ultralytics import YOLO
import cv2
import os
import time
import gc
import torch
import numpy as np
from threading import Thread, Lock
import time
from google.colab.patches import cv2_imshow


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

acci_model = YOLO("/content/drive/MyDrive/best.pt")
# yolo_model = YOLO("/content/drive/MyDrive/rd_obj_det.pt")
yolo_model = YOLO("yolov8x.pt")
tracker = "bytetrack.yaml"

# Open the video file
video_path = "/content/drive/MyDrive/Inputs/HitNRun6.mp4"
cap = cv2.VideoCapture(video_path)

new_track_path = "/content/MultipleObjectTracking_Ultralytics/HitNRun/Outputs"

output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(fps, width, height)

track_history_lock = Lock()  # Initialize a lock for track history

frame_duration = 1 / fps  # Time for one frame
accident_tracking_duration = 60  # 60 seconds
accident_tracking_frames = int(accident_tracking_duration / frame_duration)


frame_count = 0
track_history = {}
accident_participants = {}  # To store accident participants
accident_motion_history = {}
hit_and_run_cases = {}

log_file = "/content/MultipleObjectTracking_Ultralytics/HitNRun/Logs/output.txt"

def log_hit_and_run(message):
    """Log hit-and-run-related messages to a text file."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    
    print(message)

def detect_motion_change(box1, box2):
    """ Calculate instantaneous velocity based on bounding box center positions """
    x1, y1 = box1[0], box1[1]
    x2, y2 = box2[0], box2[1]
    change = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(np.floor(change))

def check_overlap(bbox1, bbox2):
    """ Check if two bounding boxes overlap """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def run_accident_detection(frame, frame_count, annotated_frame):
    # Run the accident detection model

    global accident_participants

    results = acci_model(frame, device=device)
    if results and results[0].boxes:
        log_hit_and_run(f"ACCIDENT detected in FRAME {frame_count}")
        accident_bbox_list = []
       
        for box in results[0].boxes:
            accident_bbox = box.xyxy.cpu().numpy().flatten()
            accident_bbox_list.append(accident_bbox)
           
            # Draw red bounding box for accidents
            cv2.rectangle(annotated_frame,
                          (int(accident_bbox[0]), int(accident_bbox[1])),
                          (int(accident_bbox[2]), int(accident_bbox[3])),
                          (0, 0, 255), 2)  # Red for accidents
            cv2.putText(annotated_frame, 'ACCIDENT',
                        (int(accident_bbox[0]), int(accident_bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if frame_count in track_history:
            # Check overlap with accident bbox
            overlapping_objects = set()
            for accident_bbox in accident_bbox_list:
                # Check if there are at least two objects overlapping in the accident bbox
                for track_id1, box1 in track_history[frame_count].items():
                    for track_id2, box2 in track_history[frame_count].items():
                        if track_id1 != track_id2:
                            object_bbox1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
                            object_bbox2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

                            if check_overlap(accident_bbox, object_bbox1) and check_overlap(accident_bbox, object_bbox2) and check_overlap(object_bbox1, object_bbox2):
                                overlapping_objects.update([track_id1, track_id2])

            # Track only overlapping objects (participants) for the next 5 seconds
            if len(set(overlapping_objects)) >= 2:
                accident_participants[frame_count] = list(overlapping_objects)
                start_tracking_motion(frame_count)

def start_tracking_motion(frame_count):
    """ Start tracking velocity for accident participants over accident_tracking_duration """
    accident_end_frame = frame_count + accident_tracking_frames
    
    # Iterate through the frames within accident_tracking_duration
    for f in range(frame_count, accident_end_frame):
        if f in track_history:
            for participant_id in accident_participants.get(frame_count, []):
                # Ensure we have the participant in the current frame
                if participant_id in track_history[f]:
                    # Get current and previous positions
                    current_box = track_history[f][participant_id]
                    prev_box = track_history.get(f-1, {}).get(participant_id)

                    if prev_box:
                        # Calculate velocity
                        motion_change = detect_motion_change(prev_box, current_box)

                        # If all participants have velocity = 0, it's not a hit-and-run
                        if participant_id not in accident_motion_history:
                            accident_motion_history[participant_id] = []

                        accident_motion_history[participant_id].append(motion_change)
                        log_hit_and_run(f"Motion change for participant {participant_id} in frame {f}: {motion_change}")

def evaluate_hit_and_run_cases():
    """Evaluate accumulated motion data for hit-and-run conditions at the end."""
    global hit_and_run_cases

    for participant_id, motion_changes in accident_motion_history.items():
        # Skip if participant is a person
        class_idx = None
        for f in range(frame_count, -1, -1):
            if f in track_history and participant_id in track_history[f]:
                class_idx = track_history[f][participant_id][4]
                if participant_id == 31:
                  log_hit_and_run("This class :")
                  log_hit_and_run(str(class_idx))
                break
        if class_idx is not None and class_idx == 0:
            continue  # Skip person class

        # Check if participant stopped and then fled or never stopped and left the scene
        if detect_stop_and_flee(participant_id, motion_changes):
            hit_and_run_cases[participant_id] = "stopped_then_fled"
        elif detect_failure_to_stop(participant_id, motion_changes):
            hit_and_run_cases[participant_id] = "never_stopped"

    log_hit_and_run("Final Hit-and-Run Evaluation:")
    for participant_id, reason in hit_and_run_cases.items():
        log_hit_and_run(f"Participant {participant_id} flagged for hit-and-run: {reason}")


def detect_stop_and_flee(participant_id, motion_changes):
    """
    Detect if the participant stopped at any point after the accident but then fled.
    """
    stopped_at_any_time = False
    moved_again = False

    for change in motion_changes:
        if change == 0:  # Participant stopped
            stopped_at_any_time = True
        elif change > 2 and stopped_at_any_time:  # Participant moved again
            moved_again = True

    # Hit-and-run if they stopped but then moved again (fled)
    return stopped_at_any_time and moved_again


def detect_failure_to_stop(participant_id, motion_changes):
    """
    Detect if the participant never stopped and left the scene (failure to stop after accident).
    """
    never_stopped = all(change > 1 for change in motion_changes)  # No zero velocity
    left_scene = check_if_left_scene(participant_id)

    # Hit-and-run if they never stopped and left the scene
    return never_stopped and left_scene


def check_if_left_scene(participant_id):
    """
    Check if the participant has left the field of vision.
    """
    for frame in range(frame_count + 1, accident_tracking_frames + frame_count):
        if participant_id in track_history.get(frame, {}):
            return False  # Still in the scene
    return True  # Left the scene

vehicle_classes = [0, 2, 3, 5, 7]  # Example: car, truck, bus, motorcycle
class_names = ['person','car', 'truck', 'bus', 'motorcycle']

def run_object_tracking(frame, frame_count, annotated_frame):
    # Run YOLO model for object tracking
    results = yolo_model.track(frame, device=device, persist=True, tracker=tracker, conf=0.05)

    if results and results[0].boxes and results[0].boxes.xywh is not None:
        # Get tracked boxes and their IDs
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes.xywh is not None else []
        class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []

        if len(track_ids) > 0 and len(boxes) > 0:

            for track_id, box, cls_idx in zip(track_ids, boxes, class_indices):
                # Get (x, y, w, h)

                if cls_idx in vehicle_classes:
                    x, y, w, h = box

                    # Store tracking information in track history
                    with track_history_lock:
                        if frame_count not in track_history:
                            track_history[frame_count] = {}
                        track_history[frame_count][track_id] = (x, y, w, h, cls_idx)

                    # Draw green bounding box for object tracking
                    cv2.rectangle(annotated_frame,
                                  (int(x - w / 2), int(y - h / 2)),
                                  (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 0), 2)  # Green for YOLO objects

                    # Annotate the frame with the track ID instead of the object type
                    class_name = class_names[vehicle_classes.index(cls_idx)] if cls_idx in vehicle_classes else f'Class {cls_idx}'
                    cv2.putText(annotated_frame, f'ID: {track_id}, {class_name}',
                                (int(x - w / 2), int(y - h / 2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        log_hit_and_run(f"No objects detected in frame {frame_count}")


# Main processing loop
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()

        # Run object tracking (YOLO) on the frame
        run_object_tracking(frame, frame_count, annotated_frame)

        # Run accident detection on the frame
        run_accident_detection(frame, frame_count, annotated_frame)
       
        # Display the annotated frame with both accident and object tracking
        cv2_imshow(annotated_frame)

        # Save the annotated frame to the output video
        out.write(annotated_frame)
       
        frame_count += 1
       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

evaluate_hit_and_run_cases()

# Display the accident participants results
# print("Accidents and their participants:")
# for frame_id, object_ids in accident_participants.items():
#     print(f"Frame {frame_id}: Participants IDs {object_ids}")

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

print("Saved to", new_track_path)
end = time.time()