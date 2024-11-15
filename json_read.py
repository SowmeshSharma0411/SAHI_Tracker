import os
import json

# Specify the folder containing your JSON files
folder_path = "D:\\Sowmesh\\jsons_save"
files = os.listdir(folder_path)
print(len(files))
cnt = 0
cnt_data_points = 0
max_time_stamp = 0
max_time_stamp_id = ""
max_time_stamp_file_name = ""
# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                cnt += len(data)
                for i in data:
                    if len(data[i]) > max_time_stamp:
                        max_time_stamp = len(data[i])
                        max_time_stamp_id = i
                        max_time_stamp_file_name = filename
                    cnt_data_points += len(data[i])
                print(f"{filename}: Length of dictionary is {len(data)}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}")

print(cnt, cnt_data_points, max_time_stamp,max_time_stamp_id,max_time_stamp_file_name)