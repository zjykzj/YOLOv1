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

from .dataset.cocodataset import COCODataset
from .dataset.vocdataset import VOCDataset
from .transform import Transform
from .evaluate.cocoevaluator import COCOEvaluator
from .evaluate.vocevaluator import VOCEvaluator


def build_data(cfg: Dict, data_root: str, is_train: bool = True, is_distributed: bool = False):
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
                             target_transform=None,
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
                              target_transform=None,
                              target_size=img_size,
                              max_det_nums=max_det_num
                              )
        if not is_train:
            evaluator = COCOEvaluator(dataset.coco)
    else:
        raise ValueError(f"{data_type} doesn't supports")

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg['DATA']['BATCH_SIZE'],
                                             shuffle=False,
                                             num_workers=cfg['DATA']['WORKERS'],
                                             sampler=sampler,
                                             pin_memory=True
                                             )

    return dataloader, sampler, evaluator
