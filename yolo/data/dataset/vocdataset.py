# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
from typing import Optional, List, Union
import os
import cv2
import glob
import copy
import random

import numpy as np
from numpy import ndarray

import torch
from torch.utils.data.dataset import T_co

from basedataset import BaseDataset
from ..transform import Transform
from ..target import Target
from yolo.util.box_utils import label2yolobox


class VOCDataset(BaseDataset):
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,
                 root: str,
                 name: str,
                 transform: Transform,
                 train: bool = True,
                 target_size: int = 416,
                 max_det_nums: int = 50):
        self.root = root
        self.name = name
        assert transform is not None
        self.transform = transform
        self.train = train
        self.target_size = target_size
        self.max_det_nums = max_det_nums

        image_dir = os.path.join(root, name, 'images')
        label_dir = os.path.join(root, name, 'labels')

        self.image_path_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        assert len(self.image_path_list) == len(self.label_path_list)

        self.num_classes = len(self.classes)

    def __getitem__(self, index) -> T_co:
        if self.transform.is_mosaic:
            indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
            random.shuffle(indices)
        else:
            indices = [index]

        image_list = []
        label_list = []
        for idx in indices:
            image, labels, img_path = self._load_image_label(idx)
            image_list.append(image)
            label_list.append(labels)

        image, labels, shapes = self.transform(image_list, label_list, self.target_size)

        target = self._build_target(labels, shapes, img_path)
        return image, target

    def __len__(self):
        return len(self.image_path_list)

    def _load_image_label(self, index):
        img_path = self.image_path_list[index]
        image = cv2.imread(img_path)

        label_path = self.label_path_list[index]
        # [[cls_id, x_center, y_center, box_w, box_h], ]
        # The coordinate size is relative to the width and height of the image
        labels = np.loadtxt(label_path, delimiter=' ', dtype=float)
        return image, labels, img_path

    def _build_target(self, labels: ndarray, img_info: List, img_id: Union[int, str]):
        assert isinstance(labels, ndarray)
        target = torch.zeros((self.max_det_nums, 5))
        if len(labels) > 0:
            for i, label in enumerate(labels[:self.max_det_nums]):
                target[i, :] = torch.from_numpy(label)

        if self.train:
            return target
        else:
            image_name = os.path.splitext(os.path.basename(img_id))[0]
            target = Target(target, img_info, image_name)
        return target
