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
    S = 14
    stride = 32
    arch = 'FastYOLOv1_S14'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)

    S = 7
    stride = 64
    arch = 'FastYOLOv1'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)

    shape = (1, 3, 224, 224)
    S = 7
    stride = 32
    arch = 'FastYOLOv1_S14'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)

    S = 3
    stride = 64
    arch = 'FastYOLOv1'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)


def test_yolov1():
    print('=> YOLOv1')

    shape = (1, 3, 448, 448)
    S = 14
    stride = 32
    arch = 'YOLOv1_S14'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)

    S = 7
    stride = 64
    arch = 'YOLOv1'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)

    shape = (1, 3, 224, 224)
    S = 7
    stride = 32
    arch = 'YOLOv1_S14'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)

    S = 4
    stride = 64
    arch = 'YOLOv1'
    print(f"{arch} Input: {shape} S: {S} stride: {stride}")
    data = torch.randn(shape)
    model = YOLOv1(num_classes=20, S=S, B=2, stride=stride, arch=arch, pretrained=None)

    model.train()
    outputs = model(data)
    print('Train output:', outputs.shape)

    model.eval()
    outputs = model(data)
    print('Eval output:', outputs.shape)


if __name__ == '__main__':
    test_yolov1()
    test_fastyolov1()
