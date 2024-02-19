# from PIL import Image
# import os
#
# def resize_images(directory, target_size):
#     for filename in os.listdir(directory):
#         if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more file extensions if needed
#             image_path = os.path.join(directory, filename)
#             image = Image.open(image_path)
#             s = image.size
#             resized_image = image.resize(target_size)
#             resized_image.save(image_path)
#             # print(f'resized {filename} with shape {s} to {filename} ')
#
# # Example usage
# directory_path = "/home/aous/Desktop/MIPT/project/data 3/train/mask"  # Replace with the actual directory path
# target_size = (512, 512)  # Replace with the desired target size in pixels
#
# resize_images(directory_path, target_size)

import cv2
import os

import cv2
import os

def resize_images(directory, target_size):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more file extensions if needed
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, target_size)
            cv2.imwrite(image_path, resized_image)

# Example usage
directory_path = "/home/aous/Desktop/MIPT/project/temp"  # Replace with the actual directory path
target_size = (448, 448)  # Replace with the desired target size in pixels

resize_images(directory_path, target_size)