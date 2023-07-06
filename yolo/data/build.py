# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

import os
import torch

import numpy as np

from .dataset.cocodataset import COCODataset
from .dataset.vocdataset import VOCDataset
from .evaluate.cocoevaluator import COCOEvaluator
from .evaluate.vocevaluator import VOCEvaluator
from .transform import Transform
from .target import Target


def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Data preprocess
    # [B, H, W, C] -> [B, C, H, W] -> Normalize
    images = torch.from_numpy(np.array(images, dtype=float)).permute(0, 3, 1, 2).contiguous() / 255

    if not isinstance(targets[0], Target):
        targets = torch.stack(targets)

    return images, targets


def build_data(cfg: Dict, data_root: str, is_train: bool = False, is_distributed: bool = False):
    data_type = cfg['DATA']['TYPE']
    max_det_num = cfg['DATA']['MAX_NUM_LABELS']

    sampler = None
    transform = Transform(cfg, is_train=is_train)
    dataset_name = cfg['TRAIN']['DATASET_NAME'] if is_train else cfg['TEST']['DATASET_NAME']
    img_size = cfg['TRAIN']['IMGSIZE'] if is_train else cfg['TEST']['IMGSIZE']

    evaluator = None
    if 'PASCAL VOC' == data_type:
        dataset = VOCDataset(root=data_root,
                             name=dataset_name,
                             train=is_train,
                             transform=transform,
                             target_size=img_size,
                             max_det_nums=max_det_num
                             )
        if not is_train:
            VOCdevkit_dir = os.path.join(data_root, cfg['TEST']['VOC'])
            year = cfg['TEST']['YEAR']
            split = cfg['TEST']['SPLIT']
            evaluator = VOCEvaluator(dataset.classes, VOCdevkit_dir, year=year, split=split)
    elif 'COCO' == data_type:
        dataset = COCODataset(root=data_root,
                              name=dataset_name,
                              train=is_train,
                              transform=transform,
                              target_size=img_size,
                              max_det_nums=max_det_num
                              )
        if not is_train:
            evaluator = COCOEvaluator(dataset.coco)
    else:
        raise ValueError(f"{data_type} doesn't supports")

    if is_distributed and is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg['DATA']['BATCH_SIZE'],
                                             shuffle=(sampler is None and is_train),
                                             num_workers=cfg['DATA']['WORKERS'],
                                             sampler=sampler,
                                             pin_memory=True,
                                             collate_fn=custom_collate
                                             )

    return dataloader, sampler, evaluator
