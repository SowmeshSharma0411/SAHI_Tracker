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

dir_path = "D:\\Sowmesh\\yt_vids_2\\increase_speed"

lst_files = os.listdir(dir_path)

out_dir = "D:\\Sowmesh\\yt_vids_2\\vids"
cnt = 0
for file in lst_files:
    print(cnt)
    input_video_path = os.path.join(dir_path,file)
    cap = cv2.VideoCapture(input_video_path)
    output_video_path = os.path.join(out_dir,file)
    # Get original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Keep the original frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write every second frame to increase speed by 2x
        if frame_count % 2 == 0:
            out.write(frame)
    
        frame_count += 1

    # Release everything
    cap.release()
    out.release()
    
    cnt += 1

    print("Video speed increased to 2x and saved as:", output_video_path)
    
