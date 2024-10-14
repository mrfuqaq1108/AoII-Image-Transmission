import os

start_number = 0

folder_path = 'raw-img/dataset/squirrel'

for filename in os.listdir(folder_path):
    new_filename = f"{start_number:04d}{os.path.splitext(filename)[1]}"
    old_file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, new_filename)
    os.rename(old_file_path, new_file_path)
    start_number += 1

print("complete")
