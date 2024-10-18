import os
import subprocess

def process_videos(directory_path, script_path):
    # Supported video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(directory_path, filename)
            
            # Construct the command to run the script
            command = ['python', script_path, video_path]
            
            print(f"Processing video: {filename}")
            
            # Run the script as a subprocess
            try:
                subprocess.run(command, check=True)
                print(f"Finished processing: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Directory containing the videos
    video_directory = "D:\\Sowmesh\\Test_Vids"
    
    # Path to your original script
    original_script_path = "C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\mot_ultralytics\\test24.py"
    
    # Run the processing
    process_videos(video_directory, original_script_path)