# -*- coding: utf-8 -*-

"""
@date: 2023/6/15 下午2:53
@file: cocodataset.py
@author: zj
@description: 
"""

import random
import time

import torch

from yolo.data.dataset.cocodataset import COCODataset, get_coco_label_names
from yolo.data.transform import Transform
from yolo.data.target import Target

root = '../datasets/coco'

NUM_CLASSES = 80

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


def test_train(cfg_file):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    transform = Transform(cfg, is_train=True)
    train_dataset = COCODataset(root, name='train2017', train=True, transform=transform,
                                target_size=cfg['TRAIN']['IMGSIZE'])
    print("Total len:", len(train_dataset))

    print("Train Dataset")
    end = time.time()
    for i in range(len(train_dataset)):
        image, target = train_dataset.__getitem__(i)
        image = torch.from_numpy(image)
        images = image.unsqueeze(0)
        targets = target.unsqueeze(0)

        assert_data(images, targets)
        print(f"[{i}/{len(train_dataset)}] {images.shape} {targets.shape}")
    print(f"Avg one time: {(time.time() - end) / len(train_dataset)}")


def test_val(cfg_file):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    root = '../datasets/coco'
    transform = Transform(cfg, is_train=False)
    val_dataset = COCODataset(root, name='val2017', train=False, transform=transform,
                              target_size=cfg['TEST']['IMGSIZE'])
    print("Total len:", len(val_dataset))

    end = time.time()
    for i in range(len(val_dataset)):
        image, target = val_dataset.__getitem__(i)
        image = torch.from_numpy(image)
        assert isinstance(target, Target)
        print(i, image.shape, target.target.shape, len(target.img_info), target.img_id)

        images = image.unsqueeze(0)
        targets = target.target.unsqueeze(0)
        assert_data(images, targets)
    print(f"Avg one time: {(time.time() - end) / len(val_dataset)}")

    # 获取类名
    coco_label_names, coco_class_ids, coco_cls_colors = get_coco_label_names()
    classes = [coco_label_names[index] for index in val_dataset.class_ids]
    print(classes)


if __name__ == '__main__':
    random.seed(10)

    cfg_file = 'tests/data/coco.cfg'

    print("=> COCO Train")
    test_train(cfg_file)
    print("=> COCO Val")
    test_val(cfg_file)
