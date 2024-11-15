import os 
import random
import shutil

def split_list(input_list):
    # Shuffle the input list randomly
    random.shuffle(input_list)
    
    # Calculate sizes for each split
    n = len(input_list)
    first_split = int(n * 0.5)
    second_split = int(n * 0.25)
    
    # Split the list
    list1 = input_list[:first_split]
    list2 = input_list[first_split:first_split + second_split]
    list3 = input_list[first_split + second_split:]
    
    return list1, list2, list3

# Example usage
#my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dir_path = "D:\\Sowmesh\\yt_vids_2\\v2"
lst_files = os.listdir(dir_path)
print(len(lst_files))
list1, list2, list3 = split_list(lst_files)

print("List 1 (50%)", len(list1))
print("List 2 (25%)", len(list2))
print("List 3 (25%):", len(list3))

new_dir = "D:\\Sowmesh\\yt_vids_2\\v2_new"
for file in list1:
    
    complete_path_new = os.path.join(new_dir,file)
    complete_path_old = os.path.join(dir_path,file)
    
    shutil.copy2(complete_path_old,complete_path_new)
    

    
    



