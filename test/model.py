# -*- coding: utf-8 -*-

"""
@date: 2023/5/11 下午5:14
@file: model.py
@author: zj
@description: 
"""

import torch

from yolo.model.yolov1 import YOLOv1


def test_yolov1():
    data = torch.randn(1, 3, 448, 448)
    model = YOLOv1(S=7)
    model.eval()
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 224, 224)
    model = YOLOv1(S=4)
    outputs = model(data)
    print(outputs.shape)

    data = torch.randn(1, 3, 224, 224)
    model = YOLOv1(S=4)
    outputs = model(data)
    print(outputs.shape)


if __name__ == '__main__':
    test_yolov1()
