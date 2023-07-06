# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午5:59
@file: cocodataset.py
@author: zj
@description: 
"""
import cv2
import random
import os.path

import numpy as np
from pycocotools.coco import COCO

from torch.utils.data.dataset import T_co

from .basedataset import BaseDataset
from ..transform import Transform


def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors


class COCODataset(BaseDataset):
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self,
                 root: str,
                 name: str = 'val2017',
                 train: bool = True,
                 transform: Transform = None,
                 target_size: int = 416,
                 max_det_nums=50,
                 min_size: int = 1,
                 model_type: str = 'YOLO',
                 ):
        self.root = root
        self.name = name
        self.train = train
        assert transform is not None
        self.transform = transform
        self.target_size = target_size
        # 单张图片预设的最大真值边界框数目
        self.max_det_nums = max_det_nums
        self.min_size = min_size
        self.model_type = model_type

        if 'train' in self.name:
            json_file = 'instances_train2017.json'
        elif 'val' in self.name:
            json_file = 'instances_val2017.json'
        else:
            raise ValueError(f"{name} does not match any files")
        annotation_file = os.path.join(self.root, 'annotations', json_file)
        self.coco = COCO(annotation_file)

        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.indices = range(len(self.ids))

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
        #     visualize(image, bboxes, category_ids, COCODataset.classes)
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
        # visualize(image, bboxes, category_ids, COCODataset.classes)
        # plt.show()

        target = self._build_target(labels, shapes, img_path)
        return image, target

    def __len__(self):
        return len(self.ids)

    def _load_image_label(self, index):
        # Image
        img_id = self.ids[index]
        img_path = os.path.join(self.root, 'images', self.name, '{:012}'.format(img_id) + '.jpg')
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # BBoxes
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                label = self.class_ids.index(anno['category_id'])
                x_min, y_min, box_w, box_h = anno['bbox']
                x_center = (x_min + box_w / 2.) / w
                y_center = (y_min + box_h / 2.) / h
                box_w = box_w / w
                box_h = box_h / h

                labels.append([label, x_center, y_center, box_w, box_h])
        labels = np.array(labels)

        return image, labels, img_path
