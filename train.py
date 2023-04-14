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
from yolo.data.transform import Transform


def adjust_learning_rate(base_lr, warmup_epoch,
                         optimizer: optim.Optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    # Warmup
    lr = base_lr * float(1 + step + epoch * len_epoch) / (warmup_epoch * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(base_lr, warmup_epoch,
          epoch, num_epochs, train_loader, optimizer, model, loss_fn, device):
    model.train()

    total_loss = 0.
    for i, (inputs, targets) in enumerate(train_loader):
        if (epoch + 1) <= warmup_epoch:
            adjust_learning_rate(base_lr, warmup_epoch, optimizer, epoch, i, len(train_loader))

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, targets.to(device))

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # print loss for monitoring training progress
        if i % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}, "
                f"Loss: {loss.item():.6f}, "
                f"Average Loss: {total_loss / (i + 1):.6f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def val(epoch, num_epochs, model, device, loss_fn, val_loader):
    model.eval()

    total_loss = 0.
    for i, (inputs, targets, _) in enumerate(val_loader):
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

    S = 7
    B = 2
    C = 20

    # define the YOLOv1 model
    model = YOLOv1(num_classes=C, S=S, B=B).to(device)
    print(model)

    lambda_coord = 5.
    lambda_obj = 1.
    lambda_noobj = 0.5
    lambda_class = 1.

    # define the loss function
    loss_fn = YOLOv1Loss(S=S, B=B, C=C, lambda_coord=lambda_coord, lambda_obj=lambda_obj, lambda_noobj=lambda_noobj,
                         lambda_class=lambda_class).to(device)
    print(loss_fn)

    # define the optimizer
    base_lr = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    num_epochs = 135
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 100], gamma=0.1)
    print(optimizer)
    print(lr_scheduler)

    # load the VOC dataset
    print("=> Load data")
    root = '/home/zj/yoyo/voc'
    name = 'voc2yolov5-train'
    train_dataset = VOCDataset(root, name, train=True, B=B, S=S, target_size=448, transform=Transform(is_train=True))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    name = 'voc2yolov5-val'
    val_dataset = VOCDataset(root, name, train=False, B=B, S=S, target_size=448, transform=Transform(is_train=False))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    best_epoch = 0
    best_loss = np.Inf

    warmup_epoch = 5
    # train the model
    for epoch in range(num_epochs):
        print("=> Train")
        loss = train(base_lr, warmup_epoch,
                     epoch, num_epochs, train_loader, optimizer, model, loss_fn, device)
        print(f"=> Train Loss: {loss:.6f}")
        if (epoch + 1) > warmup_epoch:
            lr_scheduler.step()

        print("=> Val")
        loss = val(epoch, num_epochs, model, device, loss_fn, val_loader)
        print(f"=> Val Loss: {loss:.6f}")

        torch.save(model.state_dict(), 'checkpoint.pth.tar')
        if loss < best_loss:
            best_epoch = epoch + 1
            best_loss = loss
            shutil.copy('checkpoint.pth.tar', 'best.pth.tar')
        print(f"=> Best Epoch: {best_epoch}\tBest loss: {best_loss:.6f}")


if __name__ == '__main__':
    main()
