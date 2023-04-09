# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
import os
import cv2
import random
from PIL import Image
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


def random_flip(im, boxes):
    if random.random() < 0.5:
        im_lr = np.fliplr(im).copy()
        # xc -> 1-xc
        boxes[:, 0] = 1 - boxes[:, 0]
        # yc -> 1-yc
        boxes[:, 1] = 1 - boxes[:, 1]

        return im_lr, boxes
    return im, boxes


def randomBlur(bgr):
    if random.random() < 0.5:
        bgr = cv2.blur(bgr, (5, 5))
    return bgr


def RandomBrightness(bgr):
    if random.random() < 0.5:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def RandomSaturation(bgr):
    if random.random() < 0.5:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def RandomHue(bgr):
    if random.random() < 0.5:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


class VOCDataset(Dataset):

    def __init__(self, root, name, train=True, transform=None, target_transform=None, B=2, S=7, target_size=448):
        self.root = root
        self.name = name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.B = B
        self.S = S
        self.target_size = target_size

        image_dir = os.path.join(root, name, 'images')
        label_dir = os.path.join(root, name, 'labels')

        self.image_path_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        assert len(self.image_path_list) == len(self.label_path_list)

        box_list = list()
        label_list = list()
        for image_path, label_path in zip(self.image_path_list, self.label_path_list):
            img_name = os.path.basename(image_path).rstrip('.jpg')
            label_name = os.path.basename(label_path).rstrip('.txt')
            assert img_name == label_name

            sub_box_list = list()
            sub_label_list = list()
            # [[cls_id, x_center, y_center, box_w, box_h], ]
            # The coordinate size is relative to the width and height of the image
            boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
            if len(boxes.shape) == 1:
                boxes = [boxes]
            for label, xc, yc, box_w, box_h in boxes:
                sub_box_list.append([xc, yc, box_w, box_h])
                sub_label_list.append(int(label))
            box_list.append(np.array(sub_box_list))
            label_list.append(sub_label_list)

        self.box_list = np.array(box_list, dtype=object)
        self.label_list = label_list
        self.num_classes = 20

        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])
        ])

    def __getitem__(self, index) -> T_co:
        image_path = self.image_path_list[index]
        boxes = self.box_list[index]
        labels = self.label_list[index]

        image = Image.open(image_path)
        if self.train:
            img = np.array(image)

            img, boxes = random_flip(img, boxes)
            img = randomBlur(img)
            img = RandomBrightness(img)
            img = RandomHue(img)
            img = RandomSaturation(img)

            image = Image.fromarray(img)

        image = self.transform(image)
        # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (self.target_size, self.target_size))

        target = self.build_target(boxes, labels)
        return image, target

    def build_target(self, boxes, labels):
        """
        :param boxes: [[xc, yc, box_w, box_h], ...]
        :param labels: [box1_cls_idx, ...]
        :return:
        """
        # [H, W, (xywh+conf)*B+num_classes]
        target = torch.zeros((self.S, self.S, 5 * self.B + self.num_classes))

        # 计算单个网格的长宽
        cell_size = 1 / self.S
        for i in range(len(boxes)):
            xc, yc, box_w, box_h = boxes[i][:4]
            x_idx, y_idx = int(xc // cell_size), int(yc // cell_size)
            if target[y_idx, x_idx, self.B * 4] == 1:
                break

            class_onehot = torch.zeros(self.num_classes)
            class_onehot[labels[i]] = 1
            # [B*5:]是分类概率
            target[y_idx, x_idx, 5 * self.B:] = class_onehot

            x_offset = xc / cell_size - x_idx
            y_offset = yc / cell_size - y_idx
            for bi in range(self.B):
                # 前B*4个是标注框
                target[y_idx, x_idx, bi * 4:(bi + 1) * 4] = \
                    torch.from_numpy(np.array([x_offset, y_offset, box_w, box_h]))
                # [B*4:B*5]是置信度
                target[y_idx, x_idx, self.B * 4 + bi] = 1.
                # target[y_idx, x_idx, bi * 5:(bi + 1) * 5] = \
                #     torch.from_numpy(np.array([x_offset, y_offset, box_w, box_h, 1]))

        return target

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    root = '/home/zj/data/voc'
    name = 'voc2yolov5-val'
    dataset = VOCDataset(root, name, S=7, B=2)
    image, target = dataset.__getitem__(300)
    print(image.shape, target.shape)
