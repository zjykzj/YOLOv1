# -*- coding: utf-8 -*-

"""
@date: 2023/4/2 下午7:53
@file: yolo.py
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


class FastYOLOv1(nn.Module):

    def __init__(self, num_classes=1000, is_expand=False):
        super(FastYOLOv1, self).__init__()

        self.num_classes = num_classes
        self.is_expand = is_expand

        self.features = nn.Sequential(
            # [1]
            conv_bn_act(3, 16, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [2]
            conv_bn_act(16, 32, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [3]
            conv_bn_act(32, 64, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [4]
            conv_bn_act(64, 128, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [5]
            conv_bn_act(128, 256, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [6]
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.Identity() if self.is_expand else nn.MaxPool2d(kernel_size=2, stride=2),

            # [7] -> [9]
            conv_bn_act(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
        )

        self.fc = nn.Sequential(
            conv_bn_act(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True,
                        act='leaky_relu'),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(1024 * self.S * self.S, 4096),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, self.num_classes)
        # )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x).reshape(-1, self.num_classes)

        return x


class YOLOv1(nn.Module):

    def __init__(self, num_classes=1000, is_expand=False):
        super(YOLOv1, self).__init__()

        self.num_classes = num_classes
        self.is_expand = is_expand

        self.features = nn.Sequential(
            # [1]
            conv_bn_act(3, 64, kernel_size=7, stride=2, padding=3, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [2]
            conv_bn_act(64, 192, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [3] -> [6]
            conv_bn_act(192, 128, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(128, 256, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 256, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(256, 512, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [7] -> [16]
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

            # [17] -> [22]
            conv_bn_act(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 512, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(512, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu') if self.is_expand else conv_bn_act(1024, 1024, kernel_size=3, stride=2,
                                                                             padding=1, bias=False, is_bn=True,
                                                                             act='leaky_relu'),

            # [23] -> [24]
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True, act='leaky_relu'),
        )

        self.fc = nn.Sequential(
            conv_bn_act(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True,
                        act='leaky_relu'),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(1024 * self.S * self.S, 4096),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, self.num_classes)
        # )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x).reshape(-1, self.num_classes)

        return x


if __name__ == '__main__':
    data = torch.randn(1, 3, 448, 448)
    model = YOLOv1(is_expand=False)
    outputs = model(data)
    print(outputs.shape)

    model = YOLOv1(is_expand=True)
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 224, 224)
    model = YOLOv1(is_expand=False)
    outputs = model(data)
    print(outputs.shape)

    model = YOLOv1(is_expand=True)
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 224, 224)
    model = YOLOv1(is_expand=False)
    outputs = model(data)
    print(outputs.shape)

    model = YOLOv1(is_expand=True)
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 448, 448)
    model = FastYOLOv1(is_expand=False)
    outputs = model(data)
    print(outputs.shape)

    model = FastYOLOv1(is_expand=True)
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 224, 224)
    model = FastYOLOv1(is_expand=False)
    outputs = model(data)
    print(outputs.shape)

    model = FastYOLOv1(is_expand=True)
    outputs = model(data)
    print(outputs.shape)
