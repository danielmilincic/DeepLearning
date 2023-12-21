import cv2
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from UNet_3Plus import UNet_3Plus
import torch
import torch.nn.functional as F
import time
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from matplotlib.ticker import EngFormatter

"""
PRELIMINARY NOTES:
Scenario 1: training without noise and testing without noise 
Scenario 2: training without noise and testing with noise
Scenario 3: training with noise and testing with noise
"""

"""
======================== HYPERPARAMETERS - START ========================
In this section, we initialize and set all the hyperparameters used in the script.
"""

class Hyperparameters:
    def __init__(self):
        """
        # HOW TO SET THE NOISE PARAMETERS:
        #
        # - for Scenario 1: set all the 3 noise parameters to 0 and SCENARIO_2 = False
        # - for Scenario 2: set the single noise parameter to the desired value and set SCENARIO_2 = True
        # - for Scenario 3: set the single noise parameter to the desired value and set SCENARIO_2 = False
        """

        self.batch_size = 8
        self.num_epochs = 1
        self.val_freq = 100
        self.learning_rate = 1e-5
        self.noise_gaussian_std = 0.00
        self.noise_salt_pepper_prob = 0
        self.noise_poisson_lambda = 0  
        self.seed = 20
        self.config = 7

    def display(self):
        print("Hyperparameters:")
        print(f"Batch size: {self.batch_size}\nNumber of epochs: {self.num_epochs}\n"
              f"Validation is done ever {self.val_freq} steps\nLearning rate: {self.learning_rate}\n"
              f"Gaussian noise standard deviation: {self.noise_gaussian_std}\n"
              f"Salt and pepper noise probability: {self.noise_salt_pepper_prob}\n"
              f"Poisson noise lambda: {self.noise_poisson_lambda}\n")

hyperparameters = Hyperparameters()

"""
======================== HYPERPARAMETERS - END ========================
"""

"""
======================== CONTROL VARIABLES - START ========================
In this section, we initialize and set all the control variables used in the script.
"""


# set to True if you want to generate the augmented dataset
GENERATION = False

# for Scenario 2 set this to True, for Scenario 1 and Scenario 3 set this to False
SCENARIO_2 = False

# set to True to plot graphs
PLOT_GRAPHS = False

# set to True to save the trained model
SAVE_MODEL = False

# set the name of the model you want to save
MODEL_TO_BE_SAVED = "model_7"

# set to True to test the model on the test set
TESTING = False

# set to True to load a previously saved model
LOAD_MODEL = False

# set the name of the model you want to load
MODEL_TO_BE_LOADED = "model_7"

# set to True to plot the accuracy over noise plots.
PLOT_NOISE_ACCURACY = False

# colour palette
DTU_BLUE = '#2f3eea'
ORANGE = '#FFAE4A'

"""
======================== CONTROL VARIABLES - END ========================"""


"""
======================== AUGMENTED DATASET GENERATION - START ========================
In this section, we generate the augmented dataset and save it to the disk."""


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

    print("Finished generating augmented data")


"""
======================== AUGMENTED DATASET GENERATION - END ========================"""

"""
======================== HELPER FUNCTIONS - START ========================
In this section, we define all the helper functions used in the script."""


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


def plot_img_label_output(org_image, ground_truth, step, name, output=None):
    """
    Converts the tensors(1,1,X,Y) to numpy arrays(X,Y) and plots them.
    :param org_image: Original image
    :param ground_truth: Ground truth
    :param step: Step in the training that is used in the plot name
    :param output: Output of the model.
    :param name: Name of the saved image
    :return: None
    """
    org_image = org_image[0].detach().cpu().squeeze().numpy()
    ground_truth = ground_truth[0].detach().cpu().squeeze().numpy()

    if output is not None:
        output = torch.argmax(output[0], dim=0).detach().cpu().squeeze().numpy()
        fig, axs = plt.subplots(1, 4)
    else:
        fig, axs = plt.subplots(1, 2)

    axs[0].imshow(org_image, cmap=coastal_breeze_cmap)
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(ground_truth, cmap=coastal_breeze_cmap)
    axs[1].set_title("Label")
    axs[1].axis('off')
    if output is not None:
        axs[2].imshow(output, cmap=coastal_breeze_cmap)
        axs[2].set_title("Prediction")
        axs[2].axis('off')
        
        difference = np.abs(ground_truth - output)
        binary_difference = np.where(difference > 0, 1, 0)
        axs[3].imshow(binary_difference, cmap='gray')
        axs[3].set_title("Error Mask")
        axs[3].axis('off')
    # create the directory if it does not exist
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.tight_layout()
    plt.savefig(f"img/conf_{hyperparameters.config}_{name}_images_{step}.png", format = 'png', bbox_inches='tight', dpi = 400)
    plt.close()


def plot_train_val_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, steps):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_acc, color=DTU_BLUE, label=' Training', linewidth=2)
    plt.plot(steps, val_acc, color=ORANGE, label='Validation', linewidth=2)
    plt.title('IOU Accuracy', fontsize='x-large')
    plt.xlabel('Step', fontsize='large')
    # plt.ylabel('IOU Accuracy', fontsize='large')
    plt.ylim([0,1])
    plt.legend(fontsize='large')  

    plt.subplot(1, 2, 2)
    plt.plot(steps, train_loss, color=DTU_BLUE, linestyle='-',  label='Training', linewidth=2)
    plt.plot(steps, val_loss, color=ORANGE, label='Validation', linewidth=2)
    plt.title('DICE Loss', fontsize='x-large')
    plt.xlabel('Step', fontsize='large')
    # plt.ylabel('DICE Loss', fontsize='large')
    plt.legend(fontsize='large')  

    plt.tight_layout()
    # create the directory if it does not exist
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(f"img/conf_{hyperparameters.config}_train_val_metric.png", format='png', bbox_inches='tight', dpi = 400)


def plot_confusion_matrix(ground_truth, predictions, step, name):
    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1, 2], normalize="true")
    class_labels = ["C0", "C1", "C2"]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels,  annot_kws={"size": 16})
    plt.xlabel('Prediction', fontsize = 'x-large')
    plt.ylabel('Label', fontsize = 'x-large')
    plt.title("Confusion Matrix", fontsize = 'x-large')
    # create the directory if it does not exist
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.tight_layout()
    if name == 'test': 
        plt.savefig(f"img/conf_{hyperparameters.config}_{name}_confmatrix.png", format='png', bbox_inches='tight', dpi = 400)
    else:
        plt.savefig(f"img/conf_{hyperparameters.config}_{name}_confmatrix_step_{step}.png", format='png', bbox_inches='tight', dpi=400)
    plt.close()


def extract_image_data(dataset):
    images_data = []
    labels_data = []
    for image, label in dataset:
        # Flatten the image and label tensors and convert them to numpy arrays
        images_data.extend(image.flatten().numpy())
        labels_data.extend(label.flatten().numpy())
    images_data = np.array(images_data)
    labels_data = np.array(labels_data)
    return images_data, labels_data


def plot_hist(data, name):
    """
    Plots the histogram of the whole data. 
    """
    data, ground_truth = extract_image_data(data)
    data_hist, data_bins = np.histogram(data, bins=256, range=(0, 1))
    ground_truth_hist, gt_bins = np.histogram(ground_truth, bins=256, range=(0, 1))

    # Create a single figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot histograms
    axs[0].hist(data_bins[:-1], data_bins, weights=data_hist, color="blue")
    axs[0].set_title(name)
    axs[1].hist(gt_bins[:-1], gt_bins, weights=ground_truth_hist, color="green")
    axs[1].set_title(f"{name} Ground Truth")

    plt.tight_layout()
    # create the directory if it does not exist
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(f"img/conf_{hyperparameters.config}_{name}_hist.png", dpi=400)
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
    min = image_in.detach().cpu().numpy().min()
    max = image_in.detach().cpu().numpy().max()
    range = max - min
    std = noise_sigma * range
    mean = 0
    gaussian = np.random.normal(mean, std, image_in.shape)
    image_in = image_in + gaussian
    image_in[image_in < 0] = 0
    image_in[image_in > 1] = 1

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
    flat_image[salt_coords] = 1
    flat_image[pepper_coords] = 0

    # Reshape the image to its original shape
    image_in = flat_image.view(image_in.shape)

    return image_in


def add_poisson_noise(image_in, lam):
    noise = np.random.poisson(lam, image_in.shape)
    image_in = image_in + noise
    image_in[image_in < 0] = 0
    image_in[image_in > 1] = 1

    return image_in


def plot_accuracy_against_noise(title, x_label, noise_values, noise_type):
    accuracies = []
    model = torch.load(f"{MODEL_TO_BE_LOADED}.pt")

    for noise in noise_values:
        test_accuracies = []
        with torch.no_grad():
            model.eval()
            for inputs, targets in test_dataloader:
                inputs = add_noise(inputs, noise, noise_type)
                inputs, targets = inputs.to(device), map_target_to_class(targets.to(device))
                output = model(inputs)
                test_accuracies.append(IOU_accuracy(targets, output))
        accuracies.append(np.mean(test_accuracies))

    plot_graph(noise_values, accuracies, title, x_label)

def add_noise(inputs, noise, noise_type):
    if noise_type == "gaussian":
        return add_gaussian_noise(inputs, noise)
    elif noise_type == "salt_and_pepper":
        return add_salt_pepper_noise(inputs, noise)

def plot_graph(x_values, y_values, title, x_label):
    plt.plot(x_values, y_values, color = DTU_BLUE, linewidth=2)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title(title, fontsize='x-large')
    plt.xlabel(x_label, fontsize='large')
    plt.ylabel('Accuracy', fontsize='large')
    plt.savefig(f"Noise_Accuracy_Plot_{title}.png", dpi = 400)
    plt.close()

"""
======================== HELPER FUNCTIONS - END ========================"""


"""
======================== DATASET LOADING - START ========================
In this section, we define the dataset class and split the data into train, validation and test sets."""


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
        # apply tensor transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label


data_directory = "new_data/"
label_directory = "new_labels/"
data = load_images_from_directory(data_directory)
labels = load_images_from_directory(label_directory)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the Coastal Breeze colormap
colors = ['#FFAE4A', '#45B7C2','#2f3eea'] 
coastal_breeze_cmap = mcolors.LinearSegmentedColormap.from_list(name='coastal_breeze_cmap', colors=colors)

tranform_toTensor = transforms.ToTensor()

# Create datasets
dataset = CustomSegmentationDataset(image_dir=data_directory, mask_dir=label_directory, transform=tranform_toTensor)

# Split data
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - (train_size + val_size)
random_seed = torch.Generator().manual_seed(hyperparameters.seed)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], random_seed)

if not LOAD_MODEL:
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparameters.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=hyperparameters.batch_size)
if TESTING:
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparameters.batch_size)

"""
======================== DATASET LOADING - END ========================"""


"""
======================== OPTIONS FOR TRAINING & TESTING  - START ========================
In this section, we set some options for training and testing."""


if SCENARIO_2:
    # Save the noise parameters in temporary variables to restore them in testing
    temp_gaussian = hyperparameters.noise_gaussian_std
    temp_salt_pepper = hyperparameters.noise_salt_pepper_prob
    temp_poisson = hyperparameters.noise_poisson_lambda

    # Set the noise parameters to 0 to not add noise during training
    hyperparameters.noise_gaussian_std = 0.00*255
    hyperparameters.noise_salt_pepper_prob = 0.00
    hyperparameters.noise_poisson_lambda = 0.00

if PLOT_NOISE_ACCURACY:
    LOAD_MODEL = True
    SCENARIO_2 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True, smooth=1.0)

"""
======================== OPTIONS FOR TRAINING & TESTING  - END ========================"""


"""
======================== TRAINING  - START ========================
In this section, we train the model."""

if not LOAD_MODEL:
    # Create model
    print("Creating Model\n")
    hyperparameters.display()
    model = UNet_3Plus(in_channels=1, n_classes=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    step = 0
    model.train()
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []

    steps = []

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

            if ((step <= 100 and step % 5 == 0) or (step % hyperparameters.val_freq == 0 and step >= 100) or step == 0 or step == 1):

                steps.append(step)

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
                    all_targets = []
                    all_predictions = []
                    for idx, (inputs, targets) in enumerate(val_dataloader):
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
                        if PLOT_GRAPHS:
                            all_targets.extend(targets.detach().cpu().numpy().flatten().tolist())
                            all_predictions.extend(torch.argmax(output, dim=1).detach().cpu().numpy().flatten().tolist())
                            if idx == len(val_dataloader)-1:
                                plot_img_label_output(inputs, targets, step, output=output, name="val")
                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches.append(IOU_accuracy(targets, output) * len(inputs))
                        valid_losses_batches.append(loss.item())
                    if PLOT_GRAPHS:
                        plot_confusion_matrix(all_targets, all_predictions, step=step, name="val")
                    model.train()

                # Append average validation accuracy to list.
                valid_accuracies.append(np.sum(valid_accuracies_batches) / len(val_dataset))
                valid_losses.append(np.sum(valid_losses_batches) / len(val_dataset))

                print(f"Step {step}  training accuracy: {train_accuracies[-1]}")
                print(f"             validation accuracy: {valid_accuracies[-1]}")
                print(f"             dice loss over the three classes: {loss}")


    current_time = time.time() - current_time
    print(f"Finished training. Took {current_time / 60} min")
    plot_train_val_loss_and_accuracy(train_losses, valid_losses, train_accuracies, valid_accuracies, steps)

    if SAVE_MODEL:
        print("Saving model")
        torch.save(model, f"{MODEL_TO_BE_SAVED}.pt")
        print("Model saved")

"""
======================== TRAINING  - END ========================"""


"""
======================== TESTING  - START ========================
In this section, we test the model."""

if TESTING and not PLOT_NOISE_ACCURACY:
    print("Testing model")
    if LOAD_MODEL:
        print("Loading model")
        model = torch.load(f"{MODEL_TO_BE_LOADED}.pt")
        print("Model loaded")
    test_accuracies = []
    test_losses = []
    with torch.no_grad():
        model.eval()
        all_targets = []
        all_predictions = []
        for idx, (inputs, targets) in enumerate(test_dataloader):
            # Add eventual noise to the inputs
            if SCENARIO_2:
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
            if PLOT_GRAPHS:
                all_targets.extend(targets.detach().cpu().numpy().flatten().tolist())
                all_predictions.extend(torch.argmax(output, dim=1).detach().cpu().numpy().flatten().tolist())
                if idx == len(test_dataloader)-1:
                    plot_img_label_output(inputs, targets, idx, output=output, name="test")


        if PLOT_GRAPHS:
            plot_confusion_matrix(all_targets, all_predictions, step=0, name="test")     
        print(f"Test accuracy: {np.mean(test_accuracies)}")
        print(f"Test loss: {np.mean(test_losses)}")
"""
======================== TESTING  - END ========================"""

"""
======================== CREATE NOISE PLOTS - START ========================
In this section, we create the noise plots."""

if PLOT_NOISE_ACCURACY:
    plot_accuracy_against_noise("Gaussian Noise", "Standard Deviation", np.linspace(0, 0.25, 25), "gaussian")
    plot_accuracy_against_noise("Salt and Pepper Noise", "Frequency", np.linspace(0, 0.025, 25), "salt_and_pepper")

"""
======================== CREATE NOISE PLOTS - END ========================
"""