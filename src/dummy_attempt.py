"""from UNet_3Plus import UNet_3Plus
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

### Give data in input to the Unet3+

'''
train_dataset = TensorDataset(torch.stack(train_images), torch.stack(train_labels))
test_dataset = TensorDataset(torch.stack(test_images), torch.stack(test_labels)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
'''

# Numero di immagini fittizie da generare
num_images = 1

# Genera dati fittizi
train_images = [torch.rand(3, 500, 500) for _ in range(num_images)]
train_labels = [torch.rand(3, 500, 500) for _ in range(num_images)]
test_images = [torch.rand(3, 500, 500) for _ in range(num_images)]
test_labels = [torch.rand(3, 500, 500) for _ in range(num_images)]

# Crea TensorDatasets
train_dataset = TensorDataset(torch.stack(train_images), torch.stack(train_labels))
test_dataset = TensorDataset(torch.stack(test_images), torch.stack(test_labels))

# Crea DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Crea il modello
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 1

# Ciclo di addestramento
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}")
"""

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
