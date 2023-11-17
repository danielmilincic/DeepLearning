import cv2
import os
import numpy as np
from sklearn import metrics
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from UNet_3Plus import UNet_3Plus
import torch.nn.functional as F
from dice_loss import *
import dice_loss as dl
import random
import time


# HYPERPARAMETERS

# Resize the images to a square of size RESIZE_TO x RESIZE_TO
RESIZE_TO = 128

# Training parameters
BATCH_SIZE = 8      # batch_size : num_steps_per_epoch => 8:44 16:22 32:11
NUM_EPOCHS = 1
VAL_EVERY_STEPS = 10
LEARNING_RATE = 1e-4

# Add Gaussian noise to the images
NOISE = True
# Add rotation and flipping to the images
ROTATION_ANGLE = 0
FLIPPING_PROBABILITY = 0.0

# Set to True to test the model after training and validation are done.
TESTING = False


def map_target_to_class(labels):
    """
    :param labels: input labels.
    :return: Returns a tensor with the classes 0,1 and 2.
    """
    return torch.round(labels * 2).squeeze(1).long()


def accuracy(target, pred):
    """
    :param target: Ground truth.
    :param pred: Output of our model.
    :return: Returns the percentage correct pixel (ground truth == model output).
    """
    map_pred = torch.argmax(pred, dim=1)
    target = target.detach().cpu().numpy()
    matches = np.sum(target == map_pred.detach().cpu().numpy())
    return matches / target.size


def plot_image_and_label_output(image, label, step,  output=None, name="example_output"):
    """
    Converts the tensors(1,1,X,Y) to numpy arrays(X,Y) and plots them.
    :param dataset: Test or train dataset
    :param idx: Index of the image and label to be plotted
    :return: None
    """
    image = image.detach().cpu().squeeze().numpy()
    label = label.detach().cpu().squeeze().numpy()
    output = output.detach().cpu().squeeze().numpy()

    if output is not None:
        fig, axs = plt.subplots(1, 3)
    else:
        fig, axs = plt.subplots(1, 2)

    axs[0].imshow(image, cmap="magma")
    axs[0].set_title("Image")
    axs[0].axis('off')
    axs[1].imshow(label, cmap="magma")
    axs[1].set_title("Label")
    axs[1].axis('off')
    if output is not None:
        axs[2].imshow(output, cmap="magma")
        axs[2].set_title("Output")
        axs[2].axis('off')
    plt.savefig(f"img/{name}_{step}.png")


def plot_train_val_loss_and_accuarcy(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("img/overfitting.png")


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
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(image_path)
        label = Image.open(label_path)

        image, label = self.apply_transform(image, label)
        return image, label

    def apply_transform(self, image, label):
        image = (np.asarray(image) / (2 ** 8 + 1)).astype(np.uint8)  # scale it to [0,255]

        # Add Gaussian noise to the image
        if NOISE:
            mean = 0
            variance = 256 # You can change this value
            sigma = np.sqrt(variance)
            gaussian = np.random.normal(mean, sigma, image.shape)
            image = image + gaussian

        image[image < 0] = 0
        image[image > 255] = 255
        image = Image.fromarray(image)

        image = self.transform(image)
        label = self.transform(label)

        return image, label

# Resize the training and validation set (501x501 -> 128x128)
transform_resized_train_val = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((RESIZE_TO, RESIZE_TO))])

# Add padding to the test set (501x501 -> 512x512)
transform_original_padded = transforms.Compose(
    [transforms.ToTensor(), transforms.Pad((6, 6, 5, 5), padding_mode="edge")
     ])

# Create datasets
dataset_train_val = CustomSegmentationDataset(image_dir="data/", mask_dir="labels/", transform=transform_resized_train_val)
dataset_test = CustomSegmentationDataset(image_dir="data/", mask_dir="labels/", transform=transform_original_padded)

# Split the first dataset into training and validation set
random_seed = torch.Generator().manual_seed(random.randint(0, 10000))
train_data, val_data = random_split(dataset_train_val, [0.75, 0.25], random_seed)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)

# Create test dataloader
test_data = dataset_test
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

# Create model
print("Creating Model ")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus(in_channels=1, n_classes=3).to(device)
# Use CrossEntropyLoss: Changes the putput of UNet3Plus from softmax to logits
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

step = 0
model.train()

train_accuracies = []
valid_accuracies = []
train_losses = []
valid_losses = []

current_time = time.time()

for epoch in range(NUM_EPOCHS):
    train_accuracies_batches = []
    train_losses_batches = []

    for inputs, targets in train_dataloader:
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        targets = map_target_to_class(targets)
        output = model(inputs)
        loss = loss_fn(output, targets)
        # if step % VAL_EVERY_STEPS == 0:
        #    print(f"loss = {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Increment step counter
        step += 1

        train_accuracies_batches.append(accuracy(targets, output))
        train_losses_batches.append(loss.item())

        if step % VAL_EVERY_STEPS == 0:

            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            train_losses.append(np.mean(train_losses_batches))

            train_accuracies_batches = []
            train_losses_batches = []

            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            valid_losses_batches = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = map_target_to_class(targets)
                    output = model(inputs)
                    # save the last image and label of the validation set
                    plot_image_and_label_output(inputs[0], targets[0], step,  torch.argmax(output[0], dim=1), name="val")
                    loss = loss_fn(output, targets)

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(accuracy(targets, output) * len(inputs))
                    valid_losses_batches.append(loss.item())
                model.train()

            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(val_data))
            valid_losses.append(np.sum(valid_losses_batches) / len(val_data))

            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            print(f"             validation accuracy: {valid_accuracies[-1]}")


if TESTING:

    # Test the mode after training and validation (model tuning) are done.
    test_accuracies_batches = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = map_target_to_class(targets)
            output = model(inputs)
            loss = loss_fn(output, targets)

            # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
            test_accuracies_batches.append(accuracy(targets, output) * len(inputs))

    # Calculate average test accuracy
    test_accuracy = np.sum(test_accuracies_batches) / len(test_data)
    print(f"Test accuracy: {test_accuracy}")


current_time = time.time() - current_time
print(f"Finished training. Took {current_time/60} min")
plot_train_val_loss_and_accuarcy(train_losses, valid_losses, train_accuracies, valid_accuracies)