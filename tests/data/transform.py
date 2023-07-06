# -*- coding: utf-8 -*-

"""
@date: 2023/7/5 下午4:25
@file: transform.py
@author: zj
@description: 
"""
import os
import cv2
import glob
import random

import numpy as np
import matplotlib.pylab as plt

from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.transform import Albumentations, augment_hsv, random_perspective, image_resize, do_mosaic, letterbox, \
    Transform
from yolo.util.plots import visualize
from yolo.util.box_utils import xywhn2xyxy, xyxy2xywhn


def get_image_label_from_voc(root, name, size=1):
    assert size >= 1

    image_dir = os.path.join(root, name, 'images')
    label_dir = os.path.join(root, name, 'labels')

    image_path_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
    assert len(image_path_list) == len(label_path_list)

    image_list = list()
    label_list = list()
    for _ in range(size):
        index = random.choice(range(len(image_path_list)))
        image_path = image_path_list[index]
        label_path = label_path_list[index]

        img_name = os.path.basename(image_path).rstrip('.jpg')
        label_name = os.path.basename(label_path).rstrip('.txt')
        assert img_name == label_name

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)

        # [[cls_id, x_center, y_center, box_w, box_h], ]
        # The coordinate size is relative to the width and height of the image
        boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
        if len(boxes.shape) == 1:
            boxes = [boxes]
        label_list.append(np.array(boxes, dtype=float))

    return image_list, label_list


def t_album():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=1)

    transform = Albumentations()
    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

        image, labels = transform(image, labels)
        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)
        plt.show()


def t_hsv():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=1)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

        augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5)
        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)
        plt.show()


def t_respective():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=3)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

        if len(labels) > 0:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h)
        image, labels = random_perspective(image,
                                           targets=labels,
                                           degrees=10,
                                           translate=.1,
                                           scale=.1,
                                           shear=10,
                                           perspective=0.0,
                                           border=(0, 0))
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=w, h=h, clip=True, eps=1E-3)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)
        plt.show()


def t_resize():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=1)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]
        print("before:", image.shape)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

        image, _, (h, w) = image_resize(image, 320)
        print("after:", image.shape)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)
        plt.show()


def t_mosaic():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=4)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]
        print("before:", image.shape)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

    target_size = 416
    mosaic_border = [-target_size // 2, -target_size // 2]
    image, labels = do_mosaic(image_list, label_list, target_size, mosaic_border)

    print("after:", image.shape)
    bboxes = []
    category_ids = []
    if len(labels) > 0:
        for label in labels:
            x0, y0, x1, y1 = label[1:]
            box_w = x1 - x0
            box_h = y1 - y0
            bboxes.append([x0, y0, box_w, box_h])
            category_ids.append(int(label[0]))

    visualize(image, bboxes, category_ids, VOCDataset.classes)

    plt.show()


def t_letterbox():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=1)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]
        print("before:", image.shape)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

        shape = 416
        image, ratio, pad = letterbox(image, shape, auto=False, scaleup=True)
        print("after:", image.shape)

        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x0, y0, x1, y1 = label[1:]
                box_w = x1 - x0
                box_h = y1 - y0
                bboxes.append([x0, y0, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)
        plt.show()


def load_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    return cfg


def t_transform_train():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=4)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]
        print("before:", image.shape)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

    cfg_file = "voc.cfg"
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")
    m = Transform(cfg, is_train=True)
    image, labels, _ = m(image_list, label_list, 416)

    h, w = image.shape[:2]
    print("after:", image.shape)
    bboxes = []
    category_ids = []
    if len(labels) > 0:
        for label in labels:
            x_c, y_c, box_w, box_h = label[1:]
            x_min = (x_c - box_w / 2) * w
            y_min = (y_c - box_h / 2) * h
            box_w = box_w * w
            box_h = box_h * h
            bboxes.append([x_min, y_min, box_w, box_h])
            category_ids.append(int(label[0]))

    visualize(image, bboxes, category_ids, VOCDataset.classes)

    plt.show()


def t_transform_val():
    image_list, label_list = get_image_label_from_voc("../../../datasets/voc/", "voc2yolov5-val", size=1)

    for image, labels in zip(image_list, label_list):
        h, w = image.shape[:2]
        print("before:", image.shape)

        bboxes = []
        category_ids = []
        if len(labels) > 0:
            for label in labels:
                x_c, y_c, box_w, box_h = label[1:]
                x_min = (x_c - box_w / 2) * w
                y_min = (y_c - box_h / 2) * h
                box_w = box_w * w
                box_h = box_h * h
                bboxes.append([x_min, y_min, box_w, box_h])
                category_ids.append(int(label[0]))

        visualize(image, bboxes, category_ids, VOCDataset.classes)

    cfg_file = "voc.cfg"
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")
    m = Transform(cfg, is_train=False)
    image, labels, _ = m(image_list, label_list, 416)

    h, w = image.shape[:2]
    print("after:", image.shape)
    bboxes = []
    category_ids = []
    if len(labels) > 0:
        for label in labels:
            x_c, y_c, box_w, box_h = label[1:]
            x_min = (x_c - box_w / 2) * w
            y_min = (y_c - box_h / 2) * h
            box_w = box_w * w
            box_h = box_h * h
            bboxes.append([x_min, y_min, box_w, box_h])
            category_ids.append(int(label[0]))

    visualize(image, bboxes, category_ids, VOCDataset.classes)

    plt.show()


if __name__ == '__main__':
    from yolo.util.logging import setup_logging

    setup_logging(0)
    # random.seed(100)

    # t_album()
    # t_hsv()
    # t_respective()
    # t_resize()
    # t_mosaic()
    # t_letterbox()

    t_transform_train()
    # t_transform_val()
