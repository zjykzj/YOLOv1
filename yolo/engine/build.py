# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午5:29
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

import time
import random

from argparse import Namespace
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from yolo.data.dataset import KEY_IMAGE_INFO, KEY_IMAGE_ID
from yolo.data.evaluate.evaluator import Evaluator
from yolo.optim.lr_schedulers.build import adjust_learning_rate
from yolo.util.metric import AverageMeter
from yolo.util.utils import postprocess, to_python_float
from yolo.util import logging

logger = logging.get_logger(__name__)


def train(args: Namespace,
          cfg: Dict,
          train_loader: DataLoader,
          model: Module,
          criterion: Module,
          optimizer: Optimizer,
          device: torch.device = None,
          epoch: int = 0):
    batch_time = AverageMeter()
    losses = AverageMeter()

    is_warmup = cfg['LR_SCHEDULER']['IS_WARMUP']
    warmup_epoch = cfg['LR_SCHEDULER']['WARMUP_EPOCH']
    accumulation_steps = cfg['TRAIN']['ACCUMULATION_STEPS']

    # switch to train mode
    model.train()
    end = time.time()

    random_resize = cfg['AUGMENTATION']['RANDOM_RESIZE']
    if random_resize:
        assert hasattr(train_loader.dataset, 'set_img_size')
        assert hasattr(train_loader.dataset, 'get_img_size')
    optimizer.zero_grad()
    for i, (input_data, target) in enumerate(train_loader):
        if is_warmup and epoch < warmup_epoch:
            adjust_learning_rate(cfg, optimizer, epoch, i, len(train_loader))

        # compute output
        output = model(input_data.to(device))
        loss = criterion(output, target.to(device)) / accumulation_steps

        # compute gradient and do SGD step
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(args, loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input_data.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            img_size = train_loader.dataset.get_img_size()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Speed {3:.3f} ({4:.3f})\t'
                        'Lr {5:.8f}\t'
                        'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                        'ImgSize: {6}x{6}\t'.format(
                (epoch + 1), (i + 1), len(train_loader),
                args.world_size * float(cfg['DATA']['BATCH_SIZE']) / batch_time.val,
                args.world_size * float(cfg['DATA']['BATCH_SIZE']) / batch_time.avg,
                current_lr,
                img_size,
                batch_time=batch_time,
                loss=losses))

            # 每隔N轮都重新指定输入图像大小
            if random_resize:
                img_size = (random.randint(0, 9) % 10 + 10) * 32
                train_loader.dataset.set_img_size(img_size)


@torch.no_grad()
def validate(val_loader: DataLoader,
             val_evaluator: Evaluator,
             model: Module,
             num_classes: int = 20,
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
        outputs = model(input_data.to(device))
        # 后处理，进行置信度阈值过滤 + NMS阈值过滤
        # 输入outputs: [B, 预测框数目, 85(xywh + obj_confg + num_classes)]
        # 输出outputs: [B, 过滤后的预测框数目, 7(xyxy + obj_conf + cls_prob + cls_id)]
        outputs = postprocess(outputs, num_classes, conf_thresh, nms_thresh)

        for i, output in enumerate(outputs):
            if output is None:
                continue

            img_info = [x[i].item() for x in targets[KEY_IMAGE_INFO]]
            img_info.append(targets[KEY_IMAGE_ID][i])

            # 提取单张图片的运行结果
            # [N_ind, 7]
            val_evaluator.put(output.cpu().data, img_info)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))

    AP50_95, AP50 = val_evaluator.result()
    return AP50_95, AP50


def reduce_tensor(args, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
