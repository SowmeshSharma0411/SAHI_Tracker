import cv2

import os

dir_path = "D:\\Sowmesh\\yt_vids_2\\part_5_new_2x"

lst_files = os.listdir(dir_path)
sec = 0
cnt = 0
def get_video_duration(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Get the total number of frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate duration in seconds
    duration = frame_count / fps if fps else 0
    
    # Release the video file
    video.release()
    
    return duration

for file in lst_files:
    if file.endswith(".mp4"):
        video_path = os.path.join(dir_path, file)
        duration = get_video_duration(video_path)
        sec += duration
        print(f"Duration of {file}: {duration:.2f} seconds")
        cnt += 1

hrs = sec//3600
mins = sec%3600//60
sec = round(sec%3600%60, -1)
if sec == 60:
    sec = 0
    mins += 1
    
    if mins == 60:
        mins = 0
        hrs += 1

print(hrs,":",mins,":",sec, " (",cnt,")")
#print(sec//3600,":",sec%3600//60,":",round(sec%3600%60, -1))