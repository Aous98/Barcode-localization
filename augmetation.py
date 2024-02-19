

''' rotation and noisy'''
# import os
# import cv2
# import numpy as np
#
# # Define the directory path of your dataset
# dataset_dir = '/home/aous/Desktop/MIPT/project/DATASET/to augment/before'
#
# # Define the directory path to save the augmented dataset
# augmented_dir = '/home/aous/Desktop/MIPT/project/DATASET/to augment/after'
#
# # Create the augmented directory if it doesn't exist
# if not os.path.exists(augmented_dir):
#     os.makedirs(augmented_dir)
#
# # Loop through each file in the dataset directory
# for filename in os.listdir(dataset_dir):
#     # Load the image
#     img_path = os.path.join(dataset_dir, filename)
#     image = cv2.imread(img_path)
#
#     # Add Gaussian noise to the image
#     noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
#     noisy_image = cv2.add(image, noise)
#
#     # Perform rotation augmentation
#     rotation_angle = np.random.randint(-10, 10)  # Adjust the range as per your requirements
#     rows, cols, _ = image.shape
#     rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
#     rotated_image = cv2.warpAffine(noisy_image, rotation_matrix, (cols, rows))
#
#     # Save the augmented image
#     augmented_filename = 'augmented_' + filename
#     augmented_path = os.path.join(augmented_dir, augmented_filename)
#     cv2.imwrite(augmented_path, rotated_image)

''' noisy'''
# import os
# import cv2
# import numpy as np
#
# # Define the directory path of your dataset
# dataset_dir = '/home/aous/Desktop/MIPT/project/DATASET/new data/temp'
#
# # Define the directory path to save the augmented dataset
# augmented_dir = '/home/aous/Desktop/MIPT/project/DATASET/new data/noisy/image'
#
# # Create the augmented directory if it doesn't exist
# if not os.path.exists(augmented_dir):
#     os.makedirs(augmented_dir)
#
# # Loop through each file in the dataset directory
# for filename in os.listdir(dataset_dir):
#     # Load the image
#     img_path = os.path.join(dataset_dir, filename)
#     image = cv2.imread(img_path)
#
#     # Add Gaussian noise to the image
#     noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
#     noisy_image = cv2.add(image, noise)
#
#     # Save the augmented image
#     augmented_filename = filename
#     augmented_path = os.path.join(augmented_dir, augmented_filename)
#     cv2.imwrite(augmented_path, noisy_image)
#     print(f"noised  {filename} to {augmented_filename}")


'''salt and peper'''
# import os
# import cv2
# import numpy as np
# import re
# # Define the directory path of your dataset
# dataset_dir = '/home/aous/Desktop/MIPT/project/DATASET/new data2/image'
#
# # Define the directory path to save the augmented dataset
# augmented_dir = '/home/aous/Desktop/MIPT/project/DATASET/new data2/noisy/image'
#
# # Create the augmented directory if it doesn't exist
# if not os.path.exists(augmented_dir):
#     os.makedirs(augmented_dir)
#
#
# # Function to add salt and pepper noise to an image
# def add_salt_and_pepper(image, salt_prob, pepper_prob):
#     noisy_image = np.copy(image)
#     salt_coords = np.random.rand(*image.shape[:2]) < salt_prob
#     pepper_coords = np.random.rand(*image.shape[:2]) < pepper_prob
#     noisy_image[salt_coords] = 255  # Add salt noise
#     noisy_image[pepper_coords] = 0  # Add pepper noise
#     return noisy_image
#
#
# # Loop through each file in the dataset directory
# for filename in os.listdir(dataset_dir):
#     # Load the image
#     img_path = os.path.join(dataset_dir, filename)
#     image = cv2.imread(img_path)
#     r = filename
#     filename = filename[:-4]
#     filename = int(filename)
#     # filename += 325
#     filename = str(filename)
#     # Add salt and pepper noise to the image
#     salt_prob = 0.02  # Adjust the probability based on the desired noise level
#     pepper_prob = 0.02  # Adjust the probability based on the desired noise level
#     noisy_image = add_salt_and_pepper(image, salt_prob, pepper_prob)
#
#     # Save the augmented image
#     # number = re.findall(r'\d+', filename)[0]
#     # number = int(number)
#     # number += 325
#     # number = str(number)
#     augmented_filename = f'{filename}.png'
#
#     augmented_path = os.path.join(augmented_dir, augmented_filename)
#     cv2.imwrite(augmented_path, noisy_image)
#     print(f"noised  {r} to {augmented_filename}")


''' resize and crop'''
# import os
# import cv2
# import numpy as np
#
# # Define the directory path of your dataset
# dataset_dir = '/home/aous/Desktop/MIPT/project/DATASET/to augment/rotated20'
#
# # Define the directory path to save the augmented dataset
# augmented_dir = '/home/aous/Desktop/MIPT/project/DATASET/to augment/rotated20'
#
# # Create the augmented directory if it doesn't exist
# if not os.path.exists(augmented_dir):
#     os.makedirs(augmented_dir)
#
# # Define the parameters for resizing and cropping
# resize_range = (0.6, 1.8)  # Range for resizing factor (randomly sampled)
# # crop_size = (256, 256)  # Size of the cropped image
# f=0
# # Loop through each file in the dataset directory
# for filename in os.listdir(dataset_dir):
#     if f==3:
#         break
#     # f+=1
#     # Load the image
#     img_path = os.path.join(dataset_dir, filename)
#     image = cv2.imread(img_path)
#
#
#     # Randomly resize the image
#     resize_factor = np.random.uniform(*resize_range)
#     if resize_factor < 1.15 and resize_factor > 0.85:
#         resize_factor -= 0.3
#     rz_bf = resize_factor
#     resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
#
#     # Randomly crop the resized image
#     h, w, _ = resized_image.shape
#     if resize_factor > 1:
#         resize_factor = 0.7
#     # print(h,w)
#     crop_size = (int(h * resize_factor),int( w * resize_factor))
#     print(h,w,crop_size, resize_factor,rz_bf)
#     top = np.random.randint(0, h - crop_size[0])
#     left = np.random.randint(0, w - crop_size[1])
#     # cropped_image = resized_image[top:top + crop_size[0], left:left + crop_size[1]]
#     cropped_image = resized_image
#     # filename = filename[:-4]
#     # filename = int(filename)
#     # r = filename
#     # filename += 650
#     # filename = str(filename)
#     # Save the augmented image
#     augmented_filename = filename # + '.jpg'
#
#     augmented_path = os.path.join(augmented_dir, augmented_filename)
#     cv2.imwrite(augmented_path, cropped_image)
#     # print(f'before{r} to {augmented_filename}')


'''rotation'''
# import os
# import cv2
# from PIL import Image
#
# # Define the source directory path where your images are located
# source_directory = '/home/aous/Desktop/MIPT/project/DATASET/to augment/before'
#
# # Define the destination directory path to save the rotated images
# destination_directory = '/home/aous/Desktop/MIPT/project/DATASET/to augment/rotated20'
# if not os.path.exists(destination_directory):
#     os.makedirs(destination_directory)
# # Specify the rotation angle in degrees
# rotation_angle = 20
#
# # Iterate over each file in the source directory
# for filename in os.listdir(source_directory):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # Read the image file
#         image_path = os.path.join(source_directory, filename)
#         image = cv2.imread(image_path)
#
#         # Get the image dimensions
#         height, width = image.shape[:2]
#
#         # Create a rotation matrix
#         rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)
#
#         # Perform the rotation
#         rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
#
#         # Determine the threshold value based on the rotation angle
#         threshold_value = 1 if rotation_angle <= 45 else 10
#
#         # Apply thresholding to handle remaining black regions
#         _, thresholded = cv2.threshold(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY), threshold_value, 255, cv2.THRESH_BINARY)
#
#         # Find contours and crop the image based on the largest contour
#         contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
#         cropped_image = rotated_image[y:y+h, x:x+w]
#
#         # Create a new filename for the rotated and cropped image
#         rotated_filename = 'rotated_' + filename
#
#         # Save the rotated and cropped image
#         cv2.imwrite(os.path.join(destination_directory, rotated_filename), cropped_image)

'''blured'''

# import os
# import cv2
#
# source_directory = '/home/aous/Desktop/MIPT/project/DATASET/new data2/temp'
#
# # Define the directory path to save the augmented dataset
# destination_directory = '/home/aous/Desktop/MIPT/project/DATASET/new data2/noisy'
# # Define the source directory path where your images are located
# # source_directory = '/home/aous/Desktop/MIPT/project/DATASET/to augment/before'
# y = 0
# # # Define the destination directory path to save the blurred images
# # destination_directory = '/home/aous/Desktop/MIPT/project/DATASET/to augment/blured'
#
# # Specify the blur kernel size
# kernel_size = (5, 5)
#
# # Iterate over each file in the source directory
# for filename in os.listdir(source_directory):
#     if y == 7:
#         break
#     # y+=1
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # Read the image file
#         image_path = os.path.join(source_directory, filename)
#         image = cv2.imread(image_path)
#         old = filename
#         # Apply blur to the image
#         blurred_image = cv2.blur(image, kernel_size)
#         filename = filename[:-4]
#         filename = int(filename)
#         # filename += 929
#         filename = str(filename)
#
#         # Create a new filename for the blurred image
#         blurred_filename = f'{filename}.png'
#
#         # Save the blurred image to the destination directory
#         cv2.imwrite(os.path.join(destination_directory, blurred_filename), blurred_image)
#         print(f'changed {old} to {blurred_filename}')
#
# #
# '''contrast and brightness'''
#
import os
import cv2

source_directory = '/home/aous/Desktop/MIPT/project/DATASET/new data2/temp'

# Define the directory path to save the augmented dataset
destination_directory = '/home/aous/Desktop/MIPT/project/DATASET/new data2/noisy'
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
# Specify the brightness and contrast adjustment values
brightness_factor = 1.2  # Increase brightness by 20%
contrast_factor = 1.5  # Increase contrast by 50%
y = 0


# Iterate over each file in the source directory
for filename in os.listdir(source_directory):
    if y == 7:
        break
    # y+=1
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image file
        image_path = os.path.join(source_directory, filename)
        image = cv2.imread(image_path)

        # Apply brightness and contrast adjustment
        adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=contrast_factor, beta=0)

        # Create a new filename for the augmented image
        augmented_filename = filename

        # Save the augmented image to the destination directory
        cv2.imwrite(os.path.join(destination_directory, augmented_filename), adjusted_image)


