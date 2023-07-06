# -*- coding: utf-8 -*-

"""
@date: 2023/5/2 下午8:31
@file: trainer.py
@author: zj
@description: 
"""
from typing import Dict

import time
import random

from argparse import Namespace

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

from yolo.optim.lr_schedulers.build import adjust_learning_rate
from yolo.util.metric import AverageMeter
from yolo.util.utils import to_python_float
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

    random_resize = cfg['AUGMENT']['RANDOM_RESIZE']
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


def reduce_tensor(args, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
