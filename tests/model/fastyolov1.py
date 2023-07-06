# -*- coding: utf-8 -*-

"""
@date: 2023/3/31 下午3:12
@file: fastyolov1.py
@author: zj
@description: 
"""

import torch.nn as nn


class FastYOLOv1_BN(nn.Module):
    def __init__(self, num_classes=20):
        super(FastYOLOv1_BN, self).__init__()
        self.num_classes = num_classes

        # Define network layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(1024)
        self.conv8 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(1024)
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * (num_classes + 5))

    def forward(self, x):
        # Implement forward pass with BN
        x = self.bn1(self.conv1(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.pool1(x)
        x = self.bn2(self.conv2(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.pool2(x)
        x = self.bn3(self.conv3(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.bn4(self.conv4(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.pool4(x)
        x = self.bn5(self.conv5(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.pool5(x)
        x = self.bn6(self.conv6(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.pool6(x)
        x = self.bn7(self.conv7(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.bn8(self.conv8(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.bn9(self.conv9(x))
        x = nn.functional.leaky_relu(x, 0.1)
        x = x.view(-1, 7 * 7 * 1024)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5)
        x = self.fc2(x)
        x = x.view(-1, 7, 7, self.num_classes + 5)
        return x
