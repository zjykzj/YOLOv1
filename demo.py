# -*- coding: utf-8 -*-

"""
@date: 2023/4/23 上午10:01
@file: demo.py
@author: zj
@description: 
"""
import copy
import glob
from typing import List, Tuple, Dict

import cv2
import yaml
import os.path

import argparse
from argparse import Namespace

import numpy as np
from numpy import ndarray

import torch.cuda
from torch import Tensor
from torch.nn import Module

from yolo.model.build import build_model
from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.dataset.cocodataset import COCODataset
from yolo.data.transform import Transform
from yolo.util.utils import postprocess
from yolo.util.box_utils import yolobox2label


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Demo.')
    parser.add_argument('cfg', type=str, default='configs/yolov2_voc.cfg', help='Model configuration file.')
    parser.add_argument('ckpt', type=str, default=None, help='Path to the checkpoint file.')
    parser.add_argument('image', type=str, default=None, help='Path to image file')

    parser.add_argument('--outputs', type=str, default='results', help='Path to save image')
    parser.add_argument('--exp', type=str, default='voc', help='Sub folder name')

    parser.add_argument('-c', '--conf-thresh', type=float, default=None, help='Confidence Threshold')
    parser.add_argument('-n', '--nms-thresh', type=float, default=None, help='NMS Threshold')
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    return args, cfg


def image_preprocess(args: Namespace, cfg: Dict):
    transform = Transform(cfg, is_train=False)
    imgsize = cfg['TEST']['IMGSIZE']

    img_path_list = list()
    img_list = list()
    img_raw_list = list()
    img_info_list = list()

    # 返回输入图像数据、原始图像数据、图像缩放前后信息
    if os.path.isfile(args.image):
        # BGR
        img_path = os.path.abspath(args.image)
        img_path_list.append(img_path)

        img = cv2.imread(args.image)
        img_raw = img.copy()

        img, _, img_info = transform(0, imgsize, img, np.array([]))
        # [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img.astype(float)).permute(2, 0, 1).contiguous() / 255
        print("img:", img.shape)

        img_list.append(img)
        img_raw_list.append(img_raw)
        img_info_list.append(img_info)
    else:
        assert os.path.isdir(args.image), args.image
        img_path_list = glob.glob(os.path.join(args.image, "*.jpg"))
        for i, img_path in enumerate(img_path_list):
            # BGR
            img = cv2.imread(img_path)
            img_raw = img.copy()

            img, _, img_info = transform(i, imgsize, img, np.array([]))
            # [H, W, C] -> [C, H, W]
            img = torch.from_numpy(img.astype(float)).permute(2, 0, 1).contiguous() / 255
            print("img:", img.shape)

            img_list.append(img)
            img_raw_list.append(img_raw)
            img_info_list.append(img_info)

    return img_list, img_raw_list, img_info_list, img_path_list


def model_init(args: Namespace, cfg: Dict):
    """
    创建模型，赋值预训练权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = build_model(args, cfg, device)

    assert args.ckpt, '--ckpt must be specified'
    if args.ckpt:
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location=device)

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, device


def parse_info(outputs: List, info_img: List or Tuple, classes: List):
    import random

    bboxes = list()
    confs = list()
    labels = list()
    colors = list()

    # x1/y1: 左上角坐标
    # x2/y2: 右下角坐标
    # conf: 置信度
    # cls_conf: 分类置信度
    # cls_pred: 分类下标
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs:
        cls_id = int(cls_pred)
        label = classes[cls_id]

        random.seed(cls_id)

        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' % (label, cls_conf.item()))
        y1, x1, y2, x2 = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append([x1, y1, x2, y2])
        labels.append(label)
        colors.append([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)])
        confs.append(conf * cls_conf)

    return bboxes, confs, labels, colors


@torch.no_grad()
def process(input_data: Tensor, model: Module, device: torch.device,
            conf_thre=0.5, nms_thre=0.45, num_classes=20):
    # img: [1, 3, 416, 416]
    # 执行模型推理，批量计算每幅图像的预测框坐标以及对应的目标置信度+分类概率
    outputs = model(input_data.unsqueeze(0).to(device)).cpu()
    # outputs: [B, N_bbox, 4(xywh)+1(conf)+num_classes]
    # 图像后处理，执行预测边界框的坐标转换以及置信度阈值过滤+NMS IoU阈值过滤
    outputs = postprocess(outputs, num_classes, conf_thre=conf_thre, nms_thre=nms_thre)

    # [B, num_det, 7]
    return outputs


def draw_bbox(img_raw: ndarray,  # 原始图像数据, BGR ndarray
              bboxes: List,  # 预测边界框
              confs: List,  # 预测边界框置信度
              labels: List,  # 预测边界框对象名
              colors: List):  # 预测边界框绘制颜色
    im = copy.deepcopy(img_raw)

    for box, conf, label, color in zip(bboxes, confs, labels, colors):
        assert len(box) == 4, box
        color = tuple([int(x) for x in color])

        # [x1, y1, x2, y2] -> [x1, y1] [x2, y2]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im, p1, p2, color, 2)

        text_str = f'{label} {conf:.3f}'
        w, h = cv2.getTextSize(text_str, 0, fontScale=0.5, thickness=1)[0]
        p1, p2 = (int(box[0]) + w, int(box[1])), (int(box[0]), int(box[1]) - h)
        cv2.rectangle(im, p1, p2, color, thickness=-1)
        org = (int(box[0]), int(box[1]))
        cv2.putText(im, text_str, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0), thickness=1)

    return im


def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    操作流程：

    1. 解析命令行参数 + 配置文件
    2. 读取图像，预处理（图像通道转换 + 图像缩放 + 数据归一化 + 维度转换 + 数据格式转换）
    3. 创建模型，加载预训练权重
    4. 模型推理 + 数据后处理（置信度阈值过滤 + NMS阈值过滤）
    5. 预测框坐标转换
    6. 预测框绘制
    """
    args, cfg = parse_args()
    print("args:", args)

    print("=> Image Prerocess")
    img_list, img_raw_list, img_info_list, img_path_list = image_preprocess(args, cfg)
    print("=> Model Init")
    model, device = model_init(args, cfg)

    print("=> Process")
    conf_thre = cfg['TEST']['CONFTHRE']
    nms_thre = cfg['TEST']['NMSTHRE']
    if args.conf_thresh:
        conf_thre = args.conf_thresh
    if args.nms_thresh:
        nms_thre = args.nms_thresh
    num_classes = cfg['MODEL']['N_CLASSES']

    data_type = cfg['DATA']['TYPE']
    if 'PASCAL VOC' == data_type:
        classes = VOCDataset.classes
    elif 'COCO' == data_type:
        classes = COCODataset.classes
    else:
        raise ValueError(f"{data_type} doesn't supports")

    save_dir = os.path.join(args.outputs, args.exp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for input_data, img_raw, img_info, img_path in zip(img_list, img_raw_list, img_info_list, img_path_list):
        print(f"Process {img_path}")
        outputs = process(input_data, model, device, conf_thre=conf_thre, nms_thre=nms_thre, num_classes=num_classes)
        if outputs[0] is None:
            print("No Objects Deteted!!")
            continue

        print("Parse INFO")
        bboxes, confs, labels, colors = parse_info(outputs[0], img_info[:6], classes)
        draw_image = draw_bbox(img_raw, bboxes, confs, labels, colors)

        img_name = os.path.basename(img_path)
        draw_image_path = os.path.join(save_dir, img_name)
        print(f"\t+ img path: {draw_image_path}")
        cv2.imwrite(draw_image_path, draw_image)


if __name__ == '__main__':
    main()
