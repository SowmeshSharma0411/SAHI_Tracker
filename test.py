from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import json
from scipy.spatial import cKDTree
from filterpy.kalman import KalmanFilter
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import torch
import gc
import threading
from sklearn.neighbors import NearestNeighbors

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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

def update_kalman(kf, x, y):
    kf.predict()
    kf.update(np.array([x, y]))
    return kf.x

def calculate_velocity(track, frame_interval, fps, frame_diagonal):
    velocities = []
    time_interval = frame_interval / fps

    for i in range(1, len(track)):
        prev_x, prev_y, prev_w, prev_h = track[i-1][:4]
        curr_x, curr_y, curr_w, curr_h = track[i][:4]
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        displacement = np.sqrt(dx**2 + dy**2)
        
        velocity = displacement / time_interval
        avg_size = (prev_w + prev_h + curr_w + curr_h) / 4
        adjusted_velocity = velocity / avg_size / frame_diagonal
        
        velocities.append((dx, dy, adjusted_velocity, curr_w, curr_h))
    
    return velocities

def estimate_average_spacing(all_objects, k=5):
    if len(all_objects) < 2:
        return 150
    
    # Extract centroids and sizes
    centroids = np.array([[obj[0], obj[1]] for obj in all_objects])
    sizes = np.array([max(obj[2], obj[3]) for obj in all_objects])
    
    # Adapt k based on the number of objects
    k = min(k, len(all_objects) - 1)
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
    distances, _ = nbrs.kneighbors(centroids)
    
    avg_distances = np.median(distances[:, 1:], axis=1)  # Exclude self-distance
    
    # Adjust spacing based on object sizes
    adjusted_spacing = avg_distances + (sizes / 4)
    
    return np.median(adjusted_spacing)

def count_nearby_objects(current_object, all_objects, frame_width, frame_height, estimated_spacing):
    # Use estimated spacing to determine dynamic radius
    dynamic_radius = estimated_spacing * 0.75  # Adjust factor as needed
    
    x, y = current_object
    x_min, x_max = max(0, x - dynamic_radius), min(frame_width, x + dynamic_radius)
    y_min, y_max = max(0, y - dynamic_radius), min(frame_height, y + dynamic_radius)
    
    # Filter objects within the search area
    nearby_objects = [obj for obj in all_objects if x_min <= obj[0] <= x_max and y_min <= obj[1] <= y_max]
    
    if not nearby_objects:
        return 0
    
    tree = cKDTree(nearby_objects)
    return len(tree.query_ball_point(current_object, dynamic_radius)) - 1

start = time.time()

# Load the YOLOv8 model
# model = YOLO("best.pt")
model = YOLO("best.pt")

# Open the video file
# video_path = "SampleVids/traffic_vid2Shortened.mp4"
# video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\D03_20240312095750.mp4"
video_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\SampleVids\\bombay_trafficShortened.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = {}
kalman_filters = {}
tracker = "bytetrack.yaml"

# Create new directory for output
new_track_dir_no = max([int(d[5:]) for d in os.listdir("runs/detect") if d.startswith("track")] + [1]) + 1
new_track_dir = f"track{new_track_dir_no}"
new_track_path = os.path.join("runs/detect", new_track_dir)
os.makedirs(new_track_path)

# Set up video writer
output_video_path = os.path.join(new_track_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(fps, width, height)

frame_diagonal = np.sqrt(width**2 + height**2)
frame_stride = max(1, int(fps / 1.5))  # Adaptive frame stride
frame_count = 0

# Vehicle classes in COCO dataset
vehicle_classes = [0,1,2,3,4,5,6,7,8,9,10,11]  # car, motorcycle, bus, truck


frame_to_calculate_distance = 360  
dynamic_radius = 150
alpha = 0.2  # Smoothing factor for dynamic radius

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, tracker=tracker)

        if not (results and results[0] and results[0].boxes):
            continue
        
        boxes = results[0].boxes.xywh
        # track_ids = results[0].boxes.id.int().tolist()
        if results[0].boxes.id is not None:  # Ensure 'id' attribute is not None
            track_ids = results[0].boxes.id.int().tolist()
        else:
            track_ids = []
        classes = results[0].boxes.cls.tolist()
        
        annotated_frame = results[0].plot()

        if frame_count == frame_to_calculate_distance:
            curr_frame_objects = [(box[0].item(), box[1].item(), box[2].item(), box[3].item()) for box, cls in zip(boxes, classes) if int(cls) in vehicle_classes]
            dynamic_radius = alpha * estimate_average_spacing(curr_frame_objects) + (1 - alpha) * dynamic_radius
        
        if frame_count % frame_stride == 0:
            for box, track_id, cls in zip(boxes, track_ids, classes):
                if int(cls) in vehicle_classes:
                    x, y, w, h = box
                    if track_id not in track_history:
                        track_history[track_id] = []
                        kalman_filters[track_id] = init_kalman_filter()
                    
                    kalman_state = update_kalman(kalman_filters[track_id], x, y)
                    smoothed_x, smoothed_y = kalman_state[0], kalman_state[1]
                    
                    track_history[track_id].append((float(smoothed_x), float(smoothed_y), float(w), float(h), int(cls)))
        
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
gc.collect()

# print(dynamic_radius)

def process_trajectory(track_id, track, all_tracks, frame_width, frame_height, vehicle_classes):
    if len(track) >= min_track_length:
        velocities = calculate_velocity(track, frame_stride, fps, frame_diagonal)
        
        nearby_objects = []
        for i, point in enumerate(track):
            # Only consider objects in the current frame
            current_frame_objects = [t[i][:2] for t in all_tracks.values() if len(t) > i and int(t[i][4]) in vehicle_classes]

            nearby = count_nearby_objects(point[:2], current_frame_objects, dynamic_radius, frame_width, frame_height)
            nearby_objects.append(nearby)
        
        return str(track_id), [
            (x, y, w, h, dx, dy, speed, nearby)
            for (x, y, w, h, _), (dx, dy, speed, _, _), nearby in zip(track[1:], velocities, nearby_objects[1:])
        ]
    return None

# Calculate velocities and nearby objects
min_track_length = 5
track_data = {}

max_threads = min(32, os.cpu_count() + 4)

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    future_to_track = {executor.submit(process_trajectory, track_id, track, track_history, width, height, vehicle_classes): track_id for track_id, track in track_history.items()}
    for future in concurrent.futures.as_completed(future_to_track):
        result = future.result()
        if result:
            track_id, track = result
            track_data[track_id] = track

    executor.shutdown(wait=True)

threading.local().__dict__.clear()
gc.collect()

# Save track data
with open(os.path.join(new_track_path, "track_history.json"), "w") as json_f:
    json.dump(track_data, json_f, indent=4)

print("Saved to", new_track_path)
end = time.time()
print("It took", end - start, "seconds!")