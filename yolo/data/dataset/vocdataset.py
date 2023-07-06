# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
import os
import cv2
import glob
import random

import numpy as np

from torch.utils.data.dataset import T_co

from .basedataset import BaseDataset
from ..transform import Transform


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
        self.indices = range(len(self.image_path_list))

    def __getitem__(self, index) -> T_co:
        if self.transform.is_mosaic:
            indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
            random.shuffle(indices)
        else:
            indices = [index]

        image_list = []
        label_list = []
        img_path = None
        for idx in indices:
            image, labels, img_path = self._load_image_label(idx)
            image_list.append(image)
            label_list.append(labels)

        #     from yolo.util.plots import visualize
        #     from matplotlib.pylab import plt
        #     bboxes = []
        #     category_ids = []
        #     h, w = image.shape[:2]
        #     for label in labels:
        #         x_c, y_c, box_w, box_h = label[1:]
        #         x_min = (x_c - box_w / 2) * w
        #         y_min = (y_c - box_h / 2) * h
        #         box_w = box_w * w
        #         box_h = box_h * h
        #         bboxes.append([x_min, y_min, box_w, box_h])
        #         category_ids.append(int(label[0]))
        #     visualize(image, bboxes, category_ids, VOCDataset.classes)
        # plt.show()
        # if not self.train:
        #     old_image = image_list[0].copy()
        #     old_labels = label_list[0].copy()

        image, labels, shapes = self.transform(image_list, label_list, self.target_size)

        # h, w = image.shape[:2]
        # print("after:", image.shape)
        # bboxes = []
        # category_ids = []
        # if len(labels) > 0:
        #     for label in labels:
        #         x_c, y_c, box_w, box_h = label[1:]
        #         x_min = (x_c - box_w / 2) * w
        #         y_min = (y_c - box_h / 2) * h
        #         box_w = box_w * w
        #         box_h = box_h * h
        #         bboxes.append([x_min, y_min, box_w, box_h])
        #         category_ids.append(int(label[0]))
        # if shapes is not None and len(labels) > 0:
        #     from yolo.util.box_utils import yolobox2label, xywhn2xyxy
        #
        #     old_labels[:, 1:] = xywhn2xyxy(old_labels[:, 1:], w=old_image.shape[1], h=old_image.shape[0])
        #     new_labels = labels.copy()
        #     new_labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w=image.shape[1], h=image.shape[0])
        #     for old_label, new_label in zip(old_labels, new_labels):
        #         x1, y1, x2, y2 = new_label[1:]
        #         y1, x1, y2, x2 = yolobox2label([y1, x1, y2, x2], shapes[:6])
        #         assert np.allclose(old_label[1:], [x1, y1, x2, y2])
        #
        # visualize(image, bboxes, category_ids, VOCDataset.classes)
        # plt.show()

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
        if len(labels.shape) == 1:
            labels = labels.reshape((1, 5))
        return image, labels, img_path
