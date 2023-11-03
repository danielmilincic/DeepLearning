from UNet_3Plus import UNet_3Plus
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

### Give data in input to the Unet3+

'''
train_dataset = TensorDataset(torch.stack(train_images), torch.stack(train_labels))
test_dataset = TensorDataset(torch.stack(test_images), torch.stack(test_labels))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_3Plus().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 10

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
