# -*- coding: utf-8 -*-

"""
@date: 2023/4/28 上午9:30
@file: val.py.py
@author: zj
@description: 
"""
from __future__ import division

import yaml
import torch.cuda

import argparse

from yolo.data.build import build_data
from yolo.engine.infer import validate
from yolo.model.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Eval.")
    parser.add_argument('data', metavar='DIR', help='Path to dataset')
    parser.add_argument('-c', '--cfg', type=str, default='configs/yolov2_voc.cfg', help='Path to config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, help='Path to checkpoint file')

    parser.add_argument('--traversal', default=False, action="store_true", help='Using different input size.')
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    print("args:", args)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    print("cfg:", cfg)

    return args, cfg


def main():
    args, cfg = parse_args()
    print("=> successfully loaded config file: ", args.cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, cfg, device=device)
    model.eval()

    if args.checkpoint:
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)

    num_classes = cfg['MODEL']['N_CLASSES']
    conf_thresh = cfg['TEST']['CONFTHRE']
    nms_thresh = float(cfg['TEST']['NMSTHRE'])

    if args.traversal:
        item_list = list(range(0, 10))
    else:
        item_list = [int(cfg['TEST']['IMGSIZE'] / 32 - 10)]

    print("=> Begin evaluating ...")
    res_list = list()

    for i in item_list:
        input_size = (i % 10 + 10) * 32
        cfg['TEST']['IMGSIZE'] = input_size
        val_loader, _, val_evaluator = build_data(cfg, args.data, is_train=False, is_distributed=False)
        # if hasattr(val_evaluator, 'save'):
        #     val_evaluator.save = True

        ap50_95, ap50 = validate(
            val_loader, val_evaluator, model,
            num_classes=num_classes, conf_thresh=conf_thresh, nms_thresh=nms_thresh, device=device)
        print(f"Input Size：[{input_size}x{input_size}] ap50_95: = {ap50_95:.4f} ap50: = {ap50:.4f}")
        res_list.append([input_size, ap50_95, ap50])

    print("=> End")
    for item in res_list:
        input_size, ap50_95, ap50 = item
        print(f"Input Size：[{input_size}x{input_size}] ap50_95: = {ap50_95:.4f} ap50: = {ap50:.4f}")


if __name__ == '__main__':
    main()
