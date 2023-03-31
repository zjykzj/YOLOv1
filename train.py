# -*- coding: utf-8 -*-

"""
@date: 2023/3/30 下午3:36
@file: ttt.py
@author: zj
@description: 
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from yolo.model.yolov1 import YOLOv1
from yolo.model.yololoss import YOLOv1Loss
from yolo.data.vocdataset import VOCDataset

device = torch.device('cpu')

# define the YOLOv1 model
model = YOLOv1().to(device)
print(model)

# define the loss function
loss_fn = YOLOv1Loss().to(device)
print(loss_fn)

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
print(optimizer)

# load the VOC dataset
print("=> Load data")
root = '/home/zj/yoyo'
name = 'yolov1-voc-train'
train_dataset = VOCDataset(root, name)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# train the model
print("=> Train")
num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, targets.to(device))

        # backward pass
        loss.backward()
        optimizer.step()

        # print loss for monitoring training progress
        # if i % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
