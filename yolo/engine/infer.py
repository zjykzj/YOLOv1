# -*- coding: utf-8 -*-

"""
@date: 2023/5/2 下午8:31
@file: infer.py
@author: zj
@description: 
"""

import time
from tqdm import tqdm

from torch.nn import Module
from torch.utils.data import DataLoader

import torch.utils.data.distributed

from yolo.data.evaluate.evaluator import Evaluator
from yolo.util.metric import AverageMeter
from yolo.util.utils import postprocess
from yolo.util import logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def validate(val_loader: DataLoader,
             val_evaluator: Evaluator,
             model: Module,
             num_classes,
             conf_thresh: float = 0.005,
             nms_thresh: float = 0.45,
             device: torch.device = None):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_data, targets) in enumerate(tqdm(val_loader)):
        # 模型推理，返回预测结果
        # img: [B, 3, 416, 416]
        outputs = model(input_data.to(device=device, dtype=torch.float))
        # 后处理，进行置信度阈值过滤 + NMS阈值过滤
        # 输入outputs: [B, 预测框数目, 85(xywh + obj_confg + num_classes)]
        # 输出outputs: [B, 过滤后的预测框数目, 7(xyxy + obj_conf + cls_prob + cls_id)]
        outputs = postprocess(outputs, num_classes, conf_thresh, nms_thresh)

        for idx, (output, target) in enumerate(zip(outputs, targets)):
            if output is None:
                continue

            img_info = target.img_info
            img_info.append(target.img_id)

            # 提取单张图片的运行结果
            # [N_ind, 7]
            val_evaluator.put(output.cpu().data, img_info)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))

    AP50_95, AP50 = val_evaluator.result()
    return AP50_95, AP50
