import os


def clear_subfolder_contents(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error: {e.strerror} - {e.filename}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except OSError as e:
                print(f"Error: {e.strerror} - {e.filename}")


# 使用示例：
directory_to_clear = 'raw-img/train'
clear_subfolder_contents(directory_to_clear)
