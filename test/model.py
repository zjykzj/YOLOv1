# -*- coding: utf-8 -*-

"""
@date: 2023/5/11 下午5:14
@file: model.py
@author: zj
@description: 
"""

import torch

from yolo.model.yolov1 import YOLOv1


def test_fastyolov1():
    print('=> FastYOLOv1')

    shape = (1, 3, 448, 448)
    print(f"Input: {shape}")
    data = torch.randn(shape)
    # model = YOLOv1(num_classes=20, S=7, B=2, arch='FastYOLOv1', pretrained=None)
    model = YOLOv1(num_classes=20, S=14, B=2, stride=32, arch='FastYOLOv1', pretrained=None)

    model.train()
    outputs = model(data)
    print(outputs.shape)

    model.eval()
    outputs = model(data)
    print(outputs.shape)

    shape = (1, 3, 224, 224)
    print(f"Input: {shape}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=7, B=2, stride=32, arch='FastYOLOv1', pretrained=None)

    model.train()
    outputs = model(data)
    print(outputs.shape)

    model.eval()
    outputs = model(data)
    print(outputs.shape)


def test_yolov1():
    print('=> YOLOv1')

    shape = (1, 3, 448, 448)
    print(f"Input: {shape}")
    data = torch.randn(shape)
    # model = YOLOv1(num_classes=20, S=7, B=2, arch='YOLOv1', pretrained=None)
    model = YOLOv1(num_classes=20, S=14, B=2, stride=32, arch='YOLOv1', pretrained=None)

    model.train()
    outputs = model(data)
    print(outputs.shape)

    model.eval()
    outputs = model(data)
    print(outputs.shape)

    shape = (1, 3, 224, 224)
    print(f"Input: {shape}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=7, B=2, stride=32, arch='YOLOv1', pretrained=None)

    model.train()
    outputs = model(data)
    print(outputs.shape)

    model.eval()
    outputs = model(data)
    print(outputs.shape)


if __name__ == '__main__':
    test_yolov1()
    test_fastyolov1()
