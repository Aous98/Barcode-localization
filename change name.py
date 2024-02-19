import os
import re

def rename_images(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    n = 6500
    y=0
    # Iterate over the files and rename them
    for file_name in files:
        if y==1:
            break
        # y+=1
        # Extract the number from the file name using regular expressions
        # number = re.findall(r'\d+', file_name)[0]
        # number = int(number)
        # number += 325
        # number = str(number)
        old = file_name
        # file_name = file_name[:-5]

        file_name = n
        # print(file_name)
        # file_name = int(file_name)
        # # file_name += 325
        # file_name = str(file_name)
        # Construct the new file name
        new_name = f"{file_name}.png"  # Change the file extension if needed

        # Get the full path of the file
        old_path = os.path.join(folder_path, old)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        n += 1
        print(f"Renamed {old} to {new_name}")

# Usage example
folder_path = '/home/aous/Desktop/MIPT/project/temp' # Replace with the actual folder path
rename_images(folder_path)


# import os
#
# def rename_images(folder_path, new_prefix):
#     # Get a list of all files in the folder
#     files = os.listdir(folder_path)
#
#     # Iterate over the files and rename them
#     for index, file_name in enumerate(files):
#         # Construct the new file name
#         new_name = f"{index}.png"  # Change the file extension if needed
#
#         # Get the full path of the file
#         old_path = os.path.join(folder_path, file_name)
#         new_path = os.path.join(folder_path_, new_name)
#
#         # Rename the file
#         os.rename(old_path, new_path)
#
#         print(f"Renamed {file_name} to {new_name}")
#
# # Usage example
# folder_path = '/home/aous/Desktop/MIPT/project/lungs_dataset/test_'
# folder_path_ = '/home/aous/Desktop/MIPT/project/lungs_dataset/test/image'  # Replace with the actual folder path
# new_prefix = ''  # Replace with the desired new prefix
# rename_images(folder_path, new_prefix)