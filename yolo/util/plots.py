# -*- coding: utf-8 -*-

"""
@date: 2023/7/5 下午4:47
@file: plots.py
@author: zj
@description:
https://albumentations.ai/docs/examples/example_bboxes/#Using-Albumentations-to-augment-bounding-boxes-for-object-detection-tasks
https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/vis.py
"""
from typing import List

import cv2
import copy

from numpy import ndarray
import matplotlib.pylab as plt

BOX_COLOR = (255, 0, 0)  # Red
# TEXT_COLOR = (255, 255, 255)  # White
TEXT_COLOR = (0, 0, 0)  # Black


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def visualize_cv2(img_raw: ndarray,  # 原始图像数据, BGR ndarray
                  bboxes: List,  # 预测边界框
                  confs: List,  # 预测边界框置信度
                  labels: List,  # 预测边界框对象名
                  colors: List):  # 预测边界框绘制颜色
    im = copy.deepcopy(img_raw)

    for box, conf, label, color in zip(bboxes, confs, labels, colors):
        assert len(box) == 4, box
        color = tuple([int(x) for x in color])

        text_str = f'{label} {conf:.3f}'
        visualize_bbox(im, box, text_str, color=color)

    return im
