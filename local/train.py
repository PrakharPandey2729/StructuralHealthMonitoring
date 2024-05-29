import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_class import CLASS_NET
from model_cover import COVER_NET
import numpy as np
import shutil
import cv2
import util
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

# Initialize models
class_net = CLASS_NET().cuda()
cover_net = COVER_NET().cuda()

batch_size = 5
learning_rate = 0.1
momentum = 0.1
num_epochs = 10
validation_split = 0.2  # Split for validation set

# Define dataset and dataloader
data_path = './Data/Train/'
transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()])
train_set = datasets.ImageFolder(data_path, transform=transform)

# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(train_set))), test_size=validation_split, random_state=42)
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=4)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(class_net.parameters()) + list(cover_net.parameters()), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust step_size based on your preference

training_losses_class = []
training_losses_cover = []
validation_losses_class = []
validation_losses_cover = []

# Training loop
for epoch in range(num_epochs):
    class_net.train()
    cover_net.train()
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.type(torch.LongTensor)
        labels = labels.cuda()

        # Forward pass
        cover_mask = cover_net(images)
        class_out = class_net(images, cover_mask)
        loss_class = criterion(class_out, labels)
        loss_cover = torch.mean(cover_mask)

        # Backward and optimize
        optimizer.zero_grad()
        loss_class.backward()
        optimizer.step()

        # Print some statistics
        print(f'Training Epoch [{epoch+1}/{num_epochs}], Loss_class: {loss_class.item()}, Loss_cover: {loss_cover.item()}')

    # Validation loop
    class_net.eval()
    cover_net.eval()
    with torch.no_grad():
        for images_val, labels_val in val_loader:
            images_val = images_val.cuda()
            labels_val = labels_val.type(torch.LongTensor)
            labels_val = labels_val.cuda()

            # Forward pass
            cover_mask_val = cover_net(images_val)
            class_out_val = class_net(images_val, cover_mask_val)
            loss_class_val = criterion(class_out_val, labels_val)
            loss_cover_val = torch.mean(cover_mask_val)

            validation_losses_class.append(loss_class_val.item())
            validation_losses_cover.append(loss_cover_val.item())

    # Adjust learning rate based on validation set
    scheduler.step(np.mean(validation_losses_class))
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'class_net_state_dict': class_net.state_dict(),
        'cover_net_state_dict': cover_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_losses_class': training_losses_class,
        'training_losses_cover': training_losses_cover,
        'validation_losses_class': validation_losses_class,
        'validation_losses_cover': validation_losses_cover,
    }
    checkpoint_path = f'checkpoint_epoch_{epoch}.pkl'
    torch.save(checkpoint, checkpoint_path)
    
    print(f'Epoch {epoch+1} - Validation Loss_class: {np.mean(validation_losses_class)}, Validation Loss_cover: {np.mean(validation_losses_cover)}')

    training_losses_class.append(np.mean(loss_class.item()))
    training_losses_cover.append(np.mean(loss_cover.item()))

# Save final models
torch.save(class_net.state_dict(), 'checkpoint/class_net_final.pth')
torch.save(cover_net.state_dict(), 'checkpoint/cover_net_final.pth')

# Plot training and validation losses
plt.plot(training_losses_class, label='Training Loss_class')
plt.plot(training_losses_cover, label='Training Loss_cover')
plt.plot(validation_losses_class, label='Validation Loss_class')
plt.plot(validation_losses_cover, label='Validation Loss_cover')
plt.legend()
plt.savefig('Model_loss.png')
