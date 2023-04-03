# -*- coding: utf-8 -*-

"""
@date: 2023/3/30 下午3:36
@file: ttt.py
@author: zj
@description: 
"""
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from yolo.model.yolov1 import YOLOv1
from yolo.model.yololoss import YOLOv1Loss
from yolo.data.vocdataset import VOCDataset


def train(epoch, num_epochs, train_loader, optimizer, model, loss_fn, device):
    model.train()

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


def val(epoch, num_epochs, model, device, loss_fn, val_loader):
    model.eval()

    total_loss = 0.
    for i, (inputs, targets) in enumerate(val_loader):
        # forward pass
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, targets.to(device))
        total_loss += loss.item()

        # print loss for monitoring training progress
        if i % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(val_loader)}], "
                f"Loss: {loss.item():.6f}, "
                f"Average Loss: {total_loss / (i + 1):.6f}")

    return total_loss / len(val_loader)


def main():
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    # define the YOLOv1 model
    model = YOLOv1().to(device)
    print(model)

    # define the loss function
    loss_fn = YOLOv1Loss().to(device)
    print(loss_fn)

    # define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.1)
    print(optimizer)
    print(lr_scheduler)

    # load the VOC dataset
    print("=> Load data")
    root = '/home/zj/yoyo'
    name = 'yolov1-voc-train'
    train_dataset = VOCDataset(root, name, train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    name = 'yolov1-voc-val'
    val_dataset = VOCDataset(root, name, train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    best_loss = np.Inf

    # train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        print("=> Train")
        train(epoch, num_epochs, train_loader, optimizer, model, loss_fn, device)
        lr_scheduler.step()

        print("=> Val")
        loss = val(epoch, num_epochs, model, device, loss_fn, val_loader)

        torch.save(model.state_dict(), 'checkpoint.pth.tar')
        if loss < best_loss:
            best_loss = loss
            print(f"=> Best loss: {best_loss:.6f}")
            shutil.copy('checkpoint.pth.tar', 'best.pth.tar')


if __name__ == '__main__':
    main()
