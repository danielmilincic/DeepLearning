import cv2

import UNet_3Plus
from imageio import imread
from PIL import Image
import os
import numpy as np
from cv2 import imread
from sklearn.model_selection import train_test_split

# model = UNet_3Plus.UNet_3Plus_DeepSup()

"""
Transform the images of the original and labeled data into arrays.
Split the data into training and test sets.
"""


def load_images_from_directory(directory):
    images = []
    image_files = sorted(os.listdir(directory))

    for image_file in image_files:
        image = cv2.imread(os.path.join(directory, image_file), cv2.IMREAD_UNCHANGED)
        images.append(image)
    return images


data_directory = "data/"
label_directory = "labels/"
data = load_images_from_directory(data_directory)
labels = load_images_from_directory(label_directory)

train_images, test_images, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=80)
