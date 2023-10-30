import cv2

import UNet_3Plus
from imageio import imread
from PIL import Image
import os
import numpy as np
from cv2 import imread
from sklearn.model_selection import train_test_split


 # model = UNet_3Plus.UNet_3Plus_DeepSup()
path_to_data="data/"
path_to_label="labels/"
data = []
labels = []


path_to_data = os.listdir(path_to_data)
tiff_data_files = [file for file in path_to_data]
tiff_data_files.sort()

path_to_label = os.listdir(path_to_label)
tiff_label_files = [file for file in path_to_label]
tiff_label_files.sort()




for data_file, label_file in zip(tiff_data_files, tiff_label_files):
    data_image = imread("data/" + data_file, cv2.IMREAD_UNCHANGED)
    data.append(data_image)
    label_image = imread("labels/" + label_file, cv2.IMREAD_UNCHANGED)
    labels.append(label_image)


train_images, test_images, train_labels, test_labels = train_test_split(data, labels,test_size=0.2,random_state=80)
print("Hi")