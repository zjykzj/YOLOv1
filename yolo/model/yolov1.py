# -*- coding: utf-8 -*-

"""
@date: 2023/3/30 上午11:14
@file: yolov1.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


def conv_bn_act(in_channels: int,
                out_channels: int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                is_bn=True,
                act='relu'):
    # 定义卷积层
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    # 定义归一化层
    if is_bn:
        norm = nn.BatchNorm2d(num_features=out_channels)
    else:
        norm = nn.Identity()

    # 定义激活层
    if 'relu' == act:
        activation = nn.ReLU(inplace=True)
    else:
        activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # 返回一个 nn.Sequential 对象，按顺序组合卷积层、归一化层和激活层
    return nn.Sequential(
        conv,
        norm,
        activation
    )


class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, S=7, B=2):
        super(YOLOv1, self).__init__()

        self.num_classes = num_classes
        self.S = S  # 特征图大小
        self.B = B  # 每个网格单元预测的边界框数量
        self.C = num_classes  # 对象类别数量

        self.features = nn.Sequential(
            conv_bn_act(3, 64, kernel_size=7, stride=2, padding=3, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_bn_act(64, 192, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_bn_act(192, 128, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(128, 256, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 256, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_bn_act(512, 256, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 256, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 256, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 256, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 512, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_bn_act(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False, is_bn=True, act='leaky_relu'),

            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)

        return x.reshape(-1, self.S, self.S, self.B * 5 + self.C)


if __name__ == '__main__':
    data = torch.randn(1, 3, 448, 448)
    model = YOLOv1(S=7)
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 224, 224)
    model = YOLOv1(S=4)
    outputs = model(data)
    print(outputs.shape)
