# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
import os

import PIL.Image
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class VOCDataset(Dataset):

    def __init__(self, root, name, transform=None, target_transform=None, B=2, S=7, target_size=224):
        self.root = root
        self.name = name
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
            boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
            if len(boxes.shape) == 1:
                boxes = [boxes]
            for label, xc, yc, box_w, box_h in boxes:
                sub_box_list.append([xc, yc, box_w, box_h])
                sub_label_list.append(label + 1)
            box_list.append(sub_box_list)
            label_list.append(sub_label_list)

        self.box_list = box_list
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

        image = PIL.Image.open(image_path)
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
            xc_yc = boxes[i][:2]
            box_wh = boxes[i][2:4]
            # 计算标注框中心点位于哪个网格
            ij_grid = np.ceil(np.array(xc_yc) / cell_size) - 1
            for bi in range(self.B):
                # 网格对应标注框置信度conf设置为1
                target[int(ij_grid[1]), int(ij_grid[0]), 5 * bi - 1 + 5] = 1
            # 指定网格的指定类别设置为1
            target[int(ij_grid[1]), int(ij_grid[0]), 5 * self.B - 1 + int(labels[i])] = 1

            # 计算中心点所在网格的长宽（相对于图像大小）
            xy_grid = ij_grid * cell_size
            # 计算中心点和对应网格左上角的偏移比例（相对于网格大小）
            frac_xy = (xc_yc - xy_grid) / cell_size
            for bi in range(self.B):
                # 网格对应标注框的真值坐标设置为
                # [
                # box_xc与对应网格左上角的偏移比例（相对于网格大小），
                # box_yc与对应网格左上角的偏移比例（相对于网格大小），
                # box_w与图像宽的比例（相对于图像宽），
                # box_h与图像高的比例（相对于图像高）
                # ]
                target[int(ij_grid[1]), int(ij_grid[0]), (5 * bi - 1 + 1):(5 * bi - 1 + 3)] = \
                    torch.from_numpy(np.array(frac_xy))
                target[int(ij_grid[1]), int(ij_grid[0]), (5 * bi - 1 + 3):(5 * bi - 1 + 5)] = \
                    torch.from_numpy(np.array(box_wh))

        return target

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    root = '/home/zj/data/voc'
    name = 'yolov1-voc-val'
    dataset = VOCDataset(root, name)
    image, target = dataset.__getitem__(300)
    print(image.shape, target.shape)