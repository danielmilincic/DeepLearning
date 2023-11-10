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


def plot_image_and_label_output(image, label, output = None):
    """
    Plots one image and its corresponding mask.
    :param dataset: Test or train dataset
    :param idx: Index of the image and label to be plotted
    :return: None
    """
    image = image.reshape(512, 512)
    label = label.reshape(512, 512)
    output = output.reshape(512, 512)

    if output is not None:
        fig, axs = plt.subplots(1, 3)
    else:
        fig, axs = plt.subplots(1, 2)

    axs[0].imshow(image, cmap="viridis")
    axs[0].set_title("Image")
    axs[0].axis('off')
    axs[1].imshow(label, cmap="viridis")
    axs[1].set_title("Label")
    axs[1].axis('off')
    if output is not None:
        axs[2].imshow(output, cmap="viridis")
        axs[2].set_title("Output")
        axs[2].axis('off')
    plt.show()


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


"""
Create a class to load our data into a dataloader and scale the u16bit and u8bit to [0,1]
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
            image = image / 65535
            min_value = image.min()
            max_value = image.max()
            image = (image - min_value) * (1 / (
                        max_value - min_value))  # scales the picture such that the minimum vlaue = 0 and the max = 1
            image = Image.fromarray(image)
            image = self.transform(image)
            label = self.transform(label)

        return image, label


"""
Transform the data. Currently resizing it to (512,512) but we could also pad or crop. 
"""
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((512, 512), antialias=True)
     ])

"""
Create dataset and split it into train and test sets. Load the test and train set into a dataloader.
"""
dataset = CustomSegmentationDataset(image_dir="data/", mask_dir="labels/", transform=transform)

random_seed = torch.Generator().manual_seed(80)
train_data, test_data, val_data = random_split(dataset, [0.8, 0.1, 0.1], random_seed)

# plot_image_and_label(train_data, 1)

train_dataloader = DataLoader(train_data, shuffle=False, batch_size=1)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

"""
Create model
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus(in_channels=1).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for idx,(image, label) in enumerate(train_dataloader):
    image = image.to(device)
    label = label.to(device)
    # Forward pass
    outputs = model(image)
    loss = criterion(outputs, label)
    if idx % 50 == 0:
        print(f'Iteration {idx}, Loss: {loss.item()}')
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plot_image_and_label_output(image, label, outputs)
    # new 
