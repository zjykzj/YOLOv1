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

device = torch.device('cuda:0')
# device = torch.device('cpu')

# define the YOLOv1 model
model = YOLOv1().to(device)
print(model)

# define the loss function
loss_fn = YOLOv1Loss().to(device)
print(loss_fn)

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.1)
print(optimizer)

# load the VOC dataset
print("=> Load data")
root = '/home/zj/yoyo'
name = 'yolov1-voc-train'
train_dataset = VOCDataset(root, name)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# train the model
print("=> Train")
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.
    for i, (inputs, targets) in enumerate(train_loader):
        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, targets.to(device))
        total_loss += loss.item()

        # backward pass
        loss.backward()
        optimizer.step()

        # print loss for monitoring training progress
        if i % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}, "
                f"Loss: {loss.item():.6f}, "
                f"Average Loss: {total_loss / (i + 1):.6f}")
    lr_scheduler.step()
