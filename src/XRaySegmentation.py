import cv2
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from UNet_3Plus import UNet_3Plus
import torch.nn.functional as F


# model = UNet_3Plus.UNet_3Plus_DeepSup()

"""
Transform the images of the original and labeled data into arrays.
Split the data into training and test sets.
"""


def get_image_files(data_path):
    return sorted(os.listdir(data_path))


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


# train_images, test_images, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=80)

"""
Create a class to load our data into a dataloader and sclae the u16bit and u8bit to [0,1]
"""


class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.label_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(self.image_dir))
        self.labels = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx]))
        label = Image.open(os.path.join(self.label_dir, self.labels[idx]))

        if self.transform is not None:
            image = np.asarray(image)
            image = image/65535
            min_value = image.min()
            max_value = image.max()
            image = (image - min_value) * (1 / (max_value - min_value)) # scales the picture such that the minimum vlaue = 0 and the max = 1
            # image = image / 65535 # scales the picture to be in [0,1]

            """"
            fig, (ax1, ax2) = plt.subplots(2)

            # Plot histograms on the first subplot
            ax1.hist(image_6, bins=500, color='blue', alpha=0.7, label='Histogram 1')
            ax1.set_ylabel('Frequency for Histogram 1')

            # Plot histograms on the second subplot
            ax2.hist(np.concatenate(image), bins=500, color='green', alpha=0.7, label='Histogram 2')
            ax2.set_ylabel('Frequency for Histogram 2')
            # Set the x-axis limits from 0 to 1
            ax1.set_xlim(0, 1)
            ax2.set_xlim(0, 1)

            # Add legends to the subplots
            ax1.legend()
            ax2.legend()
            plt.show()
            """
            image = Image.fromarray(image)
            image = self.transform(image)
            label = self.transform(label)

            image = F.pad(image, (5,6,5,6))
            label = F.pad(label, (5,6,5,6))

        return image,label


transform = transforms.Compose(
    [transforms.ToTensor()
     ])

dataset = CustomSegmentationDataset(image_dir="data/", mask_dir="labels/", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_data, shuffle=False, batch_size=2) # should be (2,1,500, 500)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus(in_channels=1).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for image, label in train_dataloader:
    # print(image, label)
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    outputs = model(image)
    loss = criterion(outputs, label)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






