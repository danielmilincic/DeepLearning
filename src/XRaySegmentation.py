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


def map_values(tensor, lower_threshold, upper_threshold):
    # Values below the lower threshold are mapped to 0
    tensor[tensor < lower_threshold] = 0

    # Values between the lower and upper thresholds are mapped to 1/2
    tensor[(tensor >= lower_threshold) & (tensor < upper_threshold)] = 0.5

    # Values equal to or above the upper threshold are mapped to 1
    tensor[tensor >= upper_threshold] = 1

    return tensor

def accuracy(target, pred):
    map_pred = map_values(pred, 0.3, 0.7)
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

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
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(image_path)
        label = Image.open(label_path)

        image, label = self.apply_transform(image, label)
        return image, label

    def apply_transform(self, image, label):
        image = np.asarray(image) / (2**16-1) # scale it to [0,1]
        image = (image - image.min()) / (image.max() - image.min()) # stretch it to include 0 and 1
        image = Image.fromarray((image * 255).astype(np.uint8)) #  convert it back to
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

# Split the data into train, validation, and test sets
random_seed = torch.Generator().manual_seed(80)
train_data, test_data, val_data = random_split(dataset, [0.8, 0.1, 0.1], random_seed)


train_dataloader = DataLoader(train_data, shuffle=False, batch_size=1)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=1)

"""
Create model
"""
print("Creating Model ")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus(in_channels=1).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 10
num_epochs = 2
validation_every_steps = 50

step = 0
model.train()

train_accuracies = []
valid_accuracies = []
        
for epoch in range(num_epochs):
    
    train_accuracies_batches = []
    
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass, compute gradients, perform one training step.
        # Your code here!
        # wrap them in Variable
        # inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward, backward, optimize
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Increment step counter
        step += 1

        #print(mapped_tensor)
        train_accuracies_batches.append(accuracy(targets, output))
        
        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            
            train_accuracies_batches = []
        
            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    loss = optimizer(output, targets)

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(accuracy(targets, output) * len(inputs))

                model.train()
                
            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(val_data))
     
            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            print(f"             test accuracy: {valid_accuracies[-1]}")

print("Finished training.")

"""
batch_size = 64
num_epochs = 20
validation_every_steps = 500

step = 0
model.train()
# Training phase
for epoch in range(num_epochs):
    model.train()
    for idx, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)

        # Forward pass
        outputs = model(image)
        loss = criterion(outputs, label)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Training Iteration {idx}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(image)
            val_loss = criterion(outputs, label)

            print(f'Validation Loss: {val_loss.item()}')


"""
"""
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
"""