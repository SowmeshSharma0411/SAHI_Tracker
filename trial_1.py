import os

# Given file path
file_path = "D:\\Sowmesh\\BACK GATE\\D24_20240325020738.mp4"

# Separate the root directory, file name, and extension
root_dir, file_name_with_ext = os.path.split(file_path)
file_name, ext = os.path.splitext(file_name_with_ext)

# Display the results
print("Root Directory:", root_dir)
print("File Name:", file_name)
print("Extension:", ext)
