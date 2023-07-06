# -*- coding: utf-8 -*-

"""
@date: 2023/6/15 下午3:50
@file: vocdataset.py
@author: zj
@description: 
"""

import time
import random

import torch

from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.transform import Transform
from yolo.data.target import Target

root = '../datasets/voc'

NUM_CLASSES = 20

W = 13
H = 13


def load_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    return cfg


def assert_data(images, targets):
    B = len(images)

    # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
    gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

    for bi in range(B):
        num_obj = gt_num_objs[bi]
        if num_obj == 0:
            continue
        # [4]: [x_c, y_c, w, h]
        gt_boxes = targets[bi][:num_obj][..., 1:]
        # [num_obj]
        gt_cls_ids = targets[bi][:num_obj][..., 0]

        gt_boxes[..., 0::2] *= W
        gt_boxes[..., 1::2] *= H

        for ni in range(num_obj):
            # [4]: [xc, yc, w, h]
            gt_box = gt_boxes[ni]
            # 对应网格下标
            cell_idx_x, cell_idx_y = torch.floor(gt_box[:2])
            assert cell_idx_x < W and cell_idx_y < H, f"{cell_idx_x} {cell_idx_y} {W} {H}"

            gt_class = gt_cls_ids[ni]
            assert int(gt_class) < NUM_CLASSES, f"{int(gt_class)} {NUM_CLASSES}"


def test_train(cfg_file, name):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    train_dataset = VOCDataset(root, name, train=True, transform=Transform(cfg, is_train=True),
                               target_size=cfg['TRAIN']['IMGSIZE'])
    print("Total len:", len(train_dataset))

    end = time.time()
    for i in range(len(train_dataset)):
        image, target = train_dataset.__getitem__(i)

        # import matplotlib.pylab as plt
        # from yolo.util.plots import visualize
        #
        # h, w = image.shape[:2]
        # print("after:", image.shape)
        # bboxes = []
        # category_ids = []
        #
        # # [num_max_det, 5] -> [num_max_det]
        # num_labels = (target.sum(dim=1) > 0).sum()
        # if num_labels > 0:
        #     for label in target[:num_labels].tolist():
        #         x_c, y_c, box_w, box_h = label[1:]
        #         x_min = (x_c - box_w / 2) * w
        #         y_min = (y_c - box_h / 2) * h
        #         box_w = box_w * w
        #         box_h = box_h * h
        #         bboxes.append([x_min, y_min, box_w, box_h])
        #         category_ids.append(int(label[0]))
        #
        # visualize(image, bboxes, category_ids, VOCDataset.classes)
        # plt.show()

        image = torch.from_numpy(image)
        images = image.unsqueeze(0)
        targets = target.unsqueeze(0)

        assert_data(images, targets)
        print(f"[{i}/{len(train_dataset)}] {images.shape} {targets.shape}")
    print(f"Avg one time: {(time.time() - end) / len(train_dataset)}")


def test_val(cfg_file, name):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    val_dataset = VOCDataset(root, name, train=False, transform=Transform(cfg, is_train=False),
                             target_size=cfg['TEST']['IMGSIZE'])
    print("Total len:", len(val_dataset))

    end = time.time()
    for i in range(len(val_dataset)):
        image, target = val_dataset.__getitem__(i)
        assert isinstance(target, Target)

        # import matplotlib.pylab as plt
        # from yolo.util.plots import visualize
        #
        # h, w = image.shape[:2]
        # print("after:", image.shape)
        # bboxes = []
        # category_ids = []
        #
        # labels = target.target.clone()
        # # [num_max_det, 5] -> [num_max_det]
        # num_labels = (labels.sum(dim=1) > 0).sum()
        # if num_labels > 0:
        #     for label in labels[:num_labels].tolist():
        #         x_c, y_c, box_w, box_h = label[1:]
        #         x_min = (x_c - box_w / 2) * w
        #         y_min = (y_c - box_h / 2) * h
        #         box_w = box_w * w
        #         box_h = box_h * h
        #         bboxes.append([x_min, y_min, box_w, box_h])
        #         category_ids.append(int(label[0]))
        #
        # visualize(image, bboxes, category_ids, VOCDataset.classes)
        # plt.show()

        image = torch.from_numpy(image)
        print(i, image.shape, target.target.shape, len(target.img_info), target.img_id)
        images = image.unsqueeze(0)
        targets = target.target.unsqueeze(0)
        assert_data(images, targets)
    print(f"Avg one time: {(time.time() - end) / len(val_dataset)}")


if __name__ == '__main__':
    random.seed(10)

    cfg_file = 'tests/data/voc.cfg'

    print("=> Pascal VOC Train")
    name = 'voc2yolov5-train'
    test_train(cfg_file, name)
    print("=> Pascal VOC Val")
    name = 'voc2yolov5-val'
    test_val(cfg_file, name)
