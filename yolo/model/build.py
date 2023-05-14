# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:00
@file: build.py
@author: zj
@description:
"""

from argparse import Namespace
from typing import Dict

import torch

from .yolov1 import YOLOv1
from .yololoss import YOLOv1Loss


def build_model(args: Namespace, cfg: Dict, device=None):
    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model_type = cfg['MODEL']['TYPE']
    if 'YOLOv1' == model_type:
        model = YOLOv1(S=cfg['MODEL']['S'],
                       B=cfg['MODEL']['B'],
                       stride=cfg['MODEL']['STRIDE'],
                       num_classes=cfg['MODEL']['N_CLASSES'],
                       arch=cfg['MODEL']['BACKBONE'],
                       pretrained=cfg['MODEL']['BACKBONE_PRETRAINED']
                       ).to(device)
    else:
        raise ValueError(f"{model_type} doesn't supports")

    model = model.to(memory_format=memory_format, device=device)
    return model


def build_criterion(cfg: Dict, device=None):
    loss_type = cfg['CRITERION']['TYPE']
    if 'YOLOv1Loss' == loss_type:
        criterion = YOLOv1Loss(S=cfg['MODEL']['S'],
                               B=cfg['MODEL']['B'],
                               C=cfg['MODEL']['N_CLASSES'],
                               lambda_coord=cfg['CRITERION']['COORD_SCALE'],
                               lambda_obj=cfg['CRITERION']['OBJ_SCALE'],
                               lambda_noobj=cfg['CRITERION']['NOOBJ_SCALE'],
                               lambda_class=cfg['CRITERION']['CLASS_SCALE'],
                               ).to(device)
    else:
        raise ValueError(f"{loss_type} doesn't supports")
    return criterion
