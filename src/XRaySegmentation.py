import cv2
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from UNet_3Plus import UNet_3Plus
import torch.nn.functional as F
from dice_loss import *
import time
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random


# HYPERPARAMETERS

class Hyperparameters:
    def __init__(self):
        """
        # INFO about Noise parameters:
        # - for Scenario 1 (no noise): set the noise parameters to 0
        # - for Scenario 2 (noise only on Test inputs): set the noise parameters to 
        #   the desired values and set NOISE_ONLY_TESTING to True
        # - for Scenario 3 (noise on Train,Val,Test inputs): set the noise parameters to 
        #   the desired values and set NOISE_ONLY_TESTING to False
        """
        self.resize_to = 128
        self.batch_size = 16
        self.num_epochs = 3
        self.val_freq = 5
        self.learning_rate = 1e-4
        self.noise_gaussian_std = 0.00*255
        self.noise_salt_pepper_prob = 0.00
        self.noise_poisson_lambda = 5  # try values around 5 maybe
        self.seed = 20
        self.config = 1

    def display(self):
        print("Hyperparameters:")
        print(f"Images resized to {self.resize_to} x {self.resize_to}")
        print(f"Batch size: {self.batch_size}\nNumber of epochs: {self.num_epochs}\n"
              f"Validation is done ever {self.val_freq} steps\nLearning rate: {self.learning_rate}\n"
              f"Gaussian noise standard deviation: {self.noise_gaussian_std}\n"
              f"Salt and pepper noise probability: {self.noise_salt_pepper_prob}\n"
              f"Poisson noise lambda: {self.noise_poisson_lambda}\n")


hyperparameters = Hyperparameters()

# DO NOT CHANGE THIS TO TRUE UNLESS YOU WANT TO GENERATE NEW DATASET FROM SCRATCH
GENERATION = True

# Set this to True if you want to test the model on the test set at the end of the training
TESTING = False

# for Scenario 2 set this to True, for Scenario 1 and Scenario 3 set this to False
NOISE_ONLY_TESTING = False


if NOISE_ONLY_TESTING:
    # Save the noise parameters in temporary variables to restore them in testing
    temp_gaussian = hyperparameters.noise_gaussian_std
    temp_salt_pepper = hyperparameters.noise_salt_pepper_prob
    temp_poisson = hyperparameters.noise_poisson_lambda

    # Set the noise parameters to 0 to not add noise during training
    hyperparameters.noise_gaussian_std = 0.00*255
    hyperparameters.noise_salt_pepper_prob = 0.00
    hyperparameters.noise_poisson_lambda = 0.00


def random_crop(image, label, crop_size):
    width, height = image.size
    random.seed(hyperparameters.seed)
    left = np.random.randint(0, width - crop_size + 1)
    top = np.random.randint(0, height - crop_size + 1)
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))

    return image, label


# Generate augmented data and save it to the disk
if GENERATION:
    crop_size = 256

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
        cropped_image, cropped_label = random_crop(hflip_image, hflip_label, crop_size)
        cropped_image.save(f"new_data/hflip_{step}.png")
        cropped_label.save(f"new_labels/hflip_{step}.png")

        # Vertical flip
        vflip_image = ImageOps.flip(image)
        vflip_label = ImageOps.flip(label)
        cropped_image, cropped_label = random_crop(vflip_image, vflip_label, crop_size)
        cropped_image.save(f"new_data/vflip_{step}.png")
        cropped_label.save(f"new_labels/vflip_{step}.png")

        # Horizontal and vertical flip
        hvflip_image = ImageOps.mirror(vflip_image)
        hvflip_label = ImageOps.mirror(vflip_label)
        cropped_image, cropped_label = random_crop(hvflip_image, hvflip_label, crop_size)
        cropped_image.save(f"new_data/hvflip_{step}.png")
        cropped_label.save(f"new_labels/hvflip_{step}.png")

        # Rotate 90 degrees
        rot90_image = image.rotate(90)
        rot90_label = label.rotate(90)
        cropped_image, cropped_label = random_crop(rot90_image, rot90_label, crop_size)
        cropped_image.save(f"new_data/rot90_{step}.png")
        cropped_label.save(f"new_labels/rot90_{step}.png")

        # Rotate 270 degrees
        rot270_image = image.rotate(270)
        rot270_label = label.rotate(270)
        cropped_image, cropped_label = random_crop(rot270_image, rot270_label, crop_size)
        cropped_image.save(f"new_data/rot270_{step}.png")
        cropped_label.save(f"new_labels/rot270_{step}.png")

        # Original image and label
        cropped_image, cropped_label = random_crop(image, label, crop_size)
        cropped_image.save(f"new_data/original_{step}.png")
        cropped_label.save(f"new_labels/original_{step}.png")

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


def iou_single_class(ground_truth, preds, class_idx):
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
    ious = [iou_single_class(ground_truth, preds, class_idx) for class_idx in range(num_classes)]
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
    if not os.path.exists(f"img/conf_{hyperparameters.config}"):
        os.mkdir(f"img/conf_{hyperparameters.config}")
    plt.savefig(f"img/conf_{hyperparameters.config}/{name}_{step}.png")
    plt.close()


def plot_train_val_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Validation Step')
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
    if not os.path.exists(f"img/conf_{hyperparameters.config}"):
        os.mkdir(f"img/conf_{hyperparameters.config}")
    plt.savefig(f"img/conf_{hyperparameters.config}/seed={hyperparameters.seed}_{hyperparameters.config}_train_val_metric.png")


def plot_confusion_matrix(ground_truth, predictions, step, name):
    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1, 2], normalize="true")
    class_labels = ["C0", "C1", "C2"]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Model Output')
    plt.ylabel('Ground Truth')
    plt.title("Confusion Matrix")
    if not os.path.exists(f"img/conf_{hyperparameters.config}"):
        os.mkdir(f"img/conf_{hyperparameters.config}")
    plt.savefig(f"img/conf_{hyperparameters.config}/step={step}_seed={hyperparameters.seed}_{hyperparameters.config}_{name}_cm.png")
    plt.close()


def extract_image_data(dataset):
    images_data = []
    labels_data = []
    for image, label in dataset:
        # Flatten the image and label tensors and convert them to numpy arrays
        images_data.extend(image.flatten().numpy())
        labels_data.extend(label.flatten().numpy())
    return images_data, labels_data


def plot_hist(data, name):
    """
    Plots the histogram of the whole data. 
    """
    data, ground_truth = extract_image_data(data)
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].hist(data, bins=256, color="blue", range=(0, 1))
    axs[0].set_title(name)
    axs[1].hist(ground_truth, bins=256, color="green", range=(0, 1))
    axs[1].set_title(f"{name} Ground Truth")
    plt.tight_layout()
    if not os.path.exists(f"img/conf_{hyperparameters.config}"):
        os.mkdir(f"img/conf_{hyperparameters.config}")
    plt.savefig(f"img/conf_{hyperparameters.config}/{name}_hist.png")
    plt.close()

def get_image_files(data_path):
    return sorted(os.listdir(data_path))


def load_images_from_directory(directory):
    images = []
    image_files = sorted(os.listdir(directory))

    for image_file in image_files:
        image = cv2.imread(os.path.join(directory, image_file), cv2.IMREAD_UNCHANGED)
        images.append(image)
    return images


def add_gaussian_noise(image_in, noise_sigma):
    mean = 0
    gaussian = np.random.normal(mean, noise_sigma, image_in.shape)
    image_in = image_in + gaussian
    image_in[image_in < 0] = 0
    image_in[image_in > 255] = 255

    # Convert image to float
    image_in = image_in.float()

    return image_in


def add_salt_pepper_noise(image_in, salt_pepper_prob):
    # Get the total number of pixels in the image
    total_pixels = image_in.numel()

    # Calculate the number of salt and pepper pixels
    num_salt = np.ceil(salt_pepper_prob * total_pixels / 2.).astype(int)
    num_pepper = np.ceil(salt_pepper_prob * total_pixels / 2.).astype(int)

    # Generate random indices for salt and pepper
    salt_coords = torch.randint(0, total_pixels, (num_salt,))
    pepper_coords = torch.randint(0, total_pixels, (num_pepper,))

    # Flatten the image for easier indexing
    flat_image = image_in.view(-1)

    # Add salt and pepper noise
    flat_image[salt_coords] = 255
    flat_image[pepper_coords] = 0

    # Reshape the image to its original shape
    image_in = flat_image.view(image_in.shape)

    # Convert image to float
    image_in = image_in.float()

    return image_in


def add_poisson_noise(image_in, lam):
    noise = np.random.poisson(lam, image_in.shape)
    image_in = image_in + noise
    image_in[image_in < 0] = 0
    image_in[image_in > 255] = 255

    # Convert image to float
    image_in = image_in.float()

    return image_in


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
        image = (np.asarray(image) / (2 ** 8 + 1)).astype(np.uint8)  # scale it to [0,255]
        image = Image.fromarray(image)
        return image, label


data_directory = "new_data/"
label_directory = "new_labels/"
data = load_images_from_directory(data_directory)
labels = load_images_from_directory(label_directory)

# Create datasets
dataset = CustomSegmentationDataset(image_dir=data_directory, mask_dir=label_directory, transform=None)

# Split data
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + val_size)
random_seed = torch.Generator().manual_seed(hyperparameters.seed)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], random_seed)

# Plot histograms
plot_hist(train_dataset, name="Train Data")
plot_hist(val_dataset, name="Validation Data")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparameters.batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=hyperparameters.batch_size)
if TESTING:
    plot_hist(test_dataset, name="Test Data")
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparameters.batch_size)

# Create model
print("Creating Model\n")
hyperparameters.display()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus(in_channels=1, n_classes=3).to(device)

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
        # Add eventual noise to the inputs
        if hyperparameters.noise_gaussian_std != 0:
            inputs = add_gaussian_noise(inputs, hyperparameters.noise_gaussian_std)
        elif hyperparameters.noise_salt_pepper_prob != 0:
            inputs = add_salt_pepper_noise(inputs, hyperparameters.noise_salt_pepper_prob)
        elif hyperparameters.noise_poisson_lambda != 0:
            inputs = add_poisson_noise(inputs, hyperparameters.noise_poisson_lambda)

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

        if step % hyperparameters.val_freq == 0 or step == 0 or step == 1:

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
                    # Add eventual noise to the inputs
                    if hyperparameters.noise_gaussian_std != 0:
                        inputs = add_gaussian_noise(inputs, hyperparameters.noise_gaussian_std)
                    elif hyperparameters.noise_salt_pepper_prob != 0:
                        inputs = add_salt_pepper_noise(inputs, hyperparameters.noise_salt_pepper_prob)
                    elif hyperparameters.noise_poisson_lambda != 0:
                        inputs = add_poisson_noise(inputs, hyperparameters.noise_poisson_lambda)

                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = map_target_to_class(targets)
                    output = model(inputs)

                    loss = loss_fn(output, targets)
                    plot_confusion_matrix(targets.detach().cpu().numpy().flatten().tolist(), torch.argmax(output, dim=1).detach().cpu().numpy().flatten().tolist(), step=step, name = "val")
                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(IOU_accuracy(targets, output) * len(inputs))
                    valid_losses_batches.append(loss.item())
                model.train()

            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(val_dataset))
            valid_losses.append(np.sum(valid_losses_batches) / len(val_dataset))

            print(f"Step {step}  training accuracy: {train_accuracies[-1]}")
            print(f"             validation accuracy: {valid_accuracies[-1]}")
            print(f"             dice loss over the three classes: {loss}")


current_time = time.time() - current_time
print(f"Finished training. Took {current_time / 60} min")
plot_train_val_loss_and_accuracy(train_losses, valid_losses, train_accuracies, valid_accuracies)

if TESTING:
    test_accuracies = []
    test_losses = []
    with torch.no_grad():
        model.eval()
        for inputs, targets in test_dataloader:
            # Add eventual noise to the inputs
            if NOISE_ONLY_TESTING:
                if temp_gaussian != 0:
                    inputs = add_gaussian_noise(inputs, temp_gaussian)
                elif temp_salt_pepper != 0:
                    inputs = add_salt_pepper_noise(inputs, temp_salt_pepper)
                elif temp_poisson != 0:
                    inputs = add_poisson_noise(inputs, temp_poisson)
            elif hyperparameters.noise_gaussian_std != 0:
                inputs = add_gaussian_noise(inputs, hyperparameters.noise_gaussian_std)
            elif hyperparameters.noise_salt_pepper_prob != 0:
                inputs = add_salt_pepper_noise(inputs, hyperparameters.noise_salt_pepper_prob)
            elif hyperparameters.noise_poisson_lambda != 0:
                inputs = add_poisson_noise(inputs, hyperparameters.noise_poisson_lambda)
            
            inputs, targets = inputs.to(device), targets.to(device)
            targets = map_target_to_class(targets)
            output = model(inputs)
            loss = loss_fn(output, targets)
            test_accuracies.append(IOU_accuracy(targets, output))
            test_losses.append(loss.item())
            plot_confusion_matrix(targets.detach().cpu().numpy().flatten().tolist(),torch.argmax(output, dim=1).detach().cpu().numpy().flatten().tolist(), step=step, name="test")

        print(f"Test accuracy: {np.mean(test_accuracies)}")
        print(f"Test loss: {np.mean(test_losses)}")