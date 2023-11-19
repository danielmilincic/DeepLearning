import cv2
import os
import numpy as np
from sklearn import metrics
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from UNet_3Plus import UNet_3Plus
import torch.nn.functional as F
from dice_loss import *
import dice_loss as dl
import random
import time
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp


# HYPERPARAMETERS

class Hyperparameters:
    def __init__(self):
        self.resize_to = 128
        self.batch_size = 1
        self.num_epochs = 1
        self.val_freq = 40
        self.learning_rate = 1e-4
        self.noise = 256

    def display(self):
        print("Hyperparameters:")
        print(f"Images resized to {self.resize_to} x {self.resize_to}")
        print(f"Batch size: {1}\nNumber of epochs: {self.num_epochs}\n"
              f"Validation is done ever {self.val_freq} steps\nLearning rate: {self.learning_rate}\nç"
              f"Noise: {self.noise}")


hyperparameters = Hyperparameters()

# DO NOT CHANGE THIS TO TRUE UNLESS YOU WANT TO GENERATE NEW DATASET FROM SCRATCH
GENERATION = False

TESTING = False

# Generate augmented data and save it to the disk
if GENERATION:

    # Create directories for the new data and labels if they do not exist
    if not os.path.exists("new_data"):
        os.mkdir("new_data")
        print("Created new_data directory")
    else:
        # Delete all files in the directory
        for file in os.listdir("new_data"):
            os.remove(f"new_data/{file}")
        print("Deleted all files in new_data directory")

    if not os.path.exists("new_labels"):
        os.mkdir("new_labels")
        print("Created new_labels directory")
    else:
        for file in os.listdir("new_labels"):
            os.remove(f"new_labels/{file}")
        print("Deleted all files in new_labels directory")

    step = 1
    # Iterate over all images and labels
    while step < 501:

        if step < 10:
            step_data = f"000{step}"
            step_label = f"00{step}"
        elif step < 100:
            step_data = f"00{step}"
            step_label = f"0{step}"
        else:
            step_data = f"0{step}"
            step_label = f"{step}"

        image = Image.open(f"data/SOCprist{step_data}.tiff")
        label = Image.open(f"labels/slice__{step_label}.tif")

        # Horizontal flip
        hflip_image = ImageOps.mirror(image)
        hflip_label = ImageOps.mirror(label)
        hflip_image.save(f"new_data/hflip_{step}.png")
        hflip_label.save(f"new_labels/hflip_{step}.png")

        # Vertical flip
        vflip_image = ImageOps.flip(image)
        vflip_label = ImageOps.flip(label)
        vflip_image.save(f"new_data/vflip_{step}.png")
        vflip_label.save(f"new_labels/vflip_{step}.png")

        # Original image and label
        image.save(f"new_data/image_{step}.png")
        label.save(f"new_labels/image_{step}.png")

        print(f"Step {step} done")
        step += 1


def map_target_to_class(labels):
    """
    :param labels: input labels.
    :return: Returns a tensor with the classes 0,1 and 2.
    """
    return torch.round(labels * 2).squeeze(1).long()


def pixelwise_accuracy(ground_truth, pred):
    """
    :param ground_truth: Ground truth
    :param pred: Output of our model
    :return: Returns the percentage correct pixel (ground truth == model output).
    """
    map_pred = torch.argmax(pred, dim=1)
    ground_truth = ground_truth.detach().cpu().numpy()
    matches = np.sum(ground_truth == map_pred.detach().cpu().numpy())
    return matches / ground_truth.size


def iou_single_class(preds, ground_truth, class_idx):
    """
    Calculates the IOU for one class
    :param preds: Output of our model
    :param ground_truth: Ground truth
    :param class_idx: Indicates the class (0,1,2, ...)
    :return: Returns the IOU for once class
    """
    class_pred = (torch.argmax(preds, dim=1) == class_idx).detach().cpu().numpy()
    class_ground_truth = (ground_truth == class_idx).detach().cpu().numpy()
    intersection = np.logical_and(class_pred, class_ground_truth).sum()
    union = np.logical_or(class_pred, class_ground_truth).sum()

    iou = intersection / union if union != 0 else 0
    return iou


def IOU_accuracy(ground_truth, preds, num_classes=3):
    """
    Calculates thee meanIOU for a given number of classes
    :param ground_truth: Ground truth
    :param preds: Output of our model
    :param num_classes: Number of different classes in the image
    """
    ious = [iou_single_class(preds, ground_truth, class_idx) for class_idx in range(num_classes)]
    mean_iou = sum(ious) / num_classes
    return mean_iou


def plot_image_and_label_output(org_image, ground_truth, step, output=None, name="example_output"):
    """
    Converts the tensors(1,1,X,Y) to numpy arrays(X,Y) and plots them.
    :param org_image: Original image
    :param ground_truth: Ground truth
    :param step: Step in the training that is used in the plot name
    :param output: Output of the model.
    :param name: Name of the saved image
    :return: None
    """
    org_image = org_image.detach().cpu().squeeze().numpy()
    ground_truth = ground_truth.detach().cpu().squeeze().numpy()
    output = output.detach().cpu().squeeze().numpy()

    if output is not None:
        fig, axs = plt.subplots(1, 3)
    else:
        fig, axs = plt.subplots(1, 2)

    axs[0].imshow(org_image, cmap="magma")
    axs[0].set_title("Image")
    axs[0].axis('off')
    axs[1].imshow(ground_truth, cmap="magma")
    axs[1].set_title("Label")
    axs[1].axis('off')
    if output is not None:
        axs[2].imshow(output, cmap="magma")
        axs[2].set_title("Output")
        axs[2].axis('off')
    # create the directory if it does not exist
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(f"img/{name}_{step}.png")


def plot_train_val_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc):
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
    # create the directory if it does not exist
    if not os.path.exists("img"):
        os.mkdir("img")
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
        if hyperparameters.noise is not None:
            mean = 0
            variance = hyperparameters.noise  # You can change this value
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
    [transforms.ToTensor(), transforms.Resize((hyperparameters.resize_to, hyperparameters.resize_to))])

# Add padding to the test set (501x501 -> 512x512)
transform_original_padded = transforms.Compose(
    [transforms.ToTensor(), transforms.Pad((6, 6, 5, 5), padding_mode="edge")
     ])

data_directory = "new_data/"
label_directory = "new_labels/"
data = load_images_from_directory(data_directory)
labels = load_images_from_directory(label_directory)

# Create datasets
dataset = CustomSegmentationDataset(image_dir=data_directory, mask_dir=label_directory,
                                    transform=None)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + val_size)

random_seed = torch.Generator().manual_seed(random.randint(0, 10000))
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], random_seed)
# Split the first dataset into training and validation set

train_dataset.dataset.transform = transform_resized_train_val
val_dataset.dataset.transform = transform_resized_train_val
test_dataset.dataset.transform = transform_original_padded

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparameters.batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=hyperparameters.batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparameters.batch_size)

# Create model
print("Creating Model ")
hyperparameters.display()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus(in_channels=1, n_classes=3).to(device)

# loss_fn = DiceLoss()
loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True, smooth=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

step = 0
model.train()

train_accuracies = []
valid_accuracies = []
train_losses = []
valid_losses = []

current_time = time.time()

for epoch in range(hyperparameters.num_epochs):
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

        train_accuracies_batches.append(IOU_accuracy(targets, output))
        train_losses_batches.append(loss.item())

        if step % hyperparameters.val_freq == 0:

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
                    # save the last image and label of the validation set plot_image_and_label_output(inputs[0],
                    # targets[0], step,  torch.argmax(output[0], dim=1), name="val")

                    loss = loss_fn(output, targets)

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(IOU_accuracy(targets, output) * len(inputs))
                    valid_losses_batches.append(loss.item())
                model.train()

            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(val_dataset))
            valid_losses.append(np.sum(valid_losses_batches) / len(val_dataset))

            print(f"Step {step}   training accuracy: {train_accuracies[-1]}")
            print(f"             validation accuracy: {valid_accuracies[-1]}")
            print(f"             dice loss over the three classes: {loss}")

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
            test_accuracies_batches.append(IOU_accuracy(targets, output) * len(inputs))

    # Calculate average test accuracy
    test_accuracy = np.sum(test_accuracies_batches) / len(test_dataset)
    print(f"Test accuracy: {test_accuracy}")

current_time = time.time() - current_time
print(f"Finished training. Took {current_time / 60} min")
plot_train_val_loss_and_accuracy(train_losses, valid_losses, train_accuracies, valid_accuracies)
