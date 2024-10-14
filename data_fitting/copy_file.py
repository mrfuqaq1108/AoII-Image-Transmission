import os
import shutil

# 设置文件夹路径
folder_path = 'raw-img/dataset'

# 获取文件夹内所有文件和文件夹的名称
filenames = os.listdir(folder_path)


def copy_files(source_folder, destination_folder1, destination_folder2, destination_folder3):
    """
    从source_folder中复制前10个文件，并将它们复制到destination_folder。
    :param source_folder: 源文件夹路径
    :param destination_folder: 目标文件夹路径
    :param file_extension: 要复制的文件的扩展名，默认为所有文件
    """
    # if not os.path.exists(destination_folder):
    #     os.makedirs(destination_folder)

    counter = 0
    for filename in os.listdir(source_folder):
        if filename.endswith('') and counter < 800:
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder1, filename)
            shutil.copy(source_path, destination_path)
            counter += 1
        elif filename.endswith('') and counter < 1100:
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder2, filename)
            shutil.copy(source_path, destination_path)
            counter += 1
        else:
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder3, filename)
            shutil.copy(source_path, destination_path)
            counter += 1


# 打印所有文件名
for file in filenames:
    source_folder = os.path.join('raw-img/dataset', file)
    destination_folder1 = os.path.join('raw-img/train', file)
    destination_folder2 = os.path.join('raw-img/valid', file)
    destination_folder3 = os.path.join('raw-img/test', file)
    copy_files(source_folder, destination_folder1, destination_folder2, destination_folder3)
