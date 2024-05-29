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

# Initialize models
class_net = CLASS_NET().cuda()
cover_net = COVER_NET().cuda()

batch_size = 1
learning_rate = 0.1
momentum = 0.9
num_epochs = 20
# Define dataset and dataloader
data_path = './Data/Train/'
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.ImageFolder(data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(class_net.parameters()) + list(cover_net.parameters()), lr=learning_rate, momentum=momentum)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.cuda()
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss_class: {loss_class.item()}, Loss_cover: {loss_cover.item()}')

# Save models if needed
torch.save(class_net.state_dict(), 'class_net.pth')
torch.save(cover_net.state_dict(), 'cover_net.pth')
