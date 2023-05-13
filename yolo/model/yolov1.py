# -*- coding: utf-8 -*-

"""
@date: 2023/3/30 上午11:14
@file: yolov1.py
@author: zj
@description: 
"""

import os

import torch
import torch.nn as nn
from torch import Tensor

from classify import yolo
from yolo.util import logging
from yolo.util.box_utils import xywh2xyxy

logger = logging.get_logger(__name__)


def conv_bn_act(in_channels: int,
                out_channels: int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                is_bn=True,
                act='relu'):
    # 定义卷积层
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    # 定义归一化层
    if is_bn:
        norm = nn.BatchNorm2d(num_features=out_channels)
    else:
        norm = nn.Identity()

    # 定义激活层
    if 'relu' == act:
        activation = nn.ReLU(inplace=True)
    elif 'leaky_relu' == act:
        activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif 'identity' == act:
        activation = nn.Identity()
    else:
        raise ValueError(f"{act} doesn't supports")

    # 返回一个 nn.Sequential 对象，按顺序组合卷积层、归一化层和激活层
    return nn.Sequential(
        conv,
        norm,
        activation
    )


class YOLOLayer(nn.Module):
    """
    YOLOLayer层操作：

    1. 获取预测框数据 / 置信度数据 / 分类数据
    2. 结合锚点框数据进行预测框坐标转换
    """

    stride = 64

    def __init__(self, num_classes=20, S=7, B=2):
        super(YOLOLayer, self).__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B

    def forward(self, outputs: Tensor):
        N, n_ch, H, W = outputs.shape[:4]
        assert n_ch == (5 * self.B + self.num_classes)
        assert H == W == self.S

        dtype = outputs.dtype
        device = outputs.device

        # grid coordinate
        # [W] -> [1, 1, W, 1] -> [N, H, W, B]
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, W, 1),
                                     (N, H, W, self.B)).to(dtype=dtype, device=device)
        # [H] -> [1, H, 1, 1] -> [N, H, W, B]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, H, 1, 1),
                                     (N, H, W, self.B)).to(dtype=dtype, device=device)

        # [N, n_ch, H, W] -> [N, H, W, n_ch]
        outputs = outputs.permute(0, 2, 3, 1)
        # x/y/w/h/conf/probs compress to [0,1]
        outputs = torch.sigmoid(outputs)

        # [N, H, W, B*4] -> [N, H, W, B, 4]
        pred_boxes = outputs[..., :(self.B * 4)].reshape(N, H, W, self.B, 4)
        # [N, H, W, B]
        pred_confs = outputs[..., (self.B * 4):(self.B * 5)]
        # [N, H, W, num_classes]
        pred_probs = outputs[..., (self.B * 5):]

        preds = torch.zeros(N, H, W, self.B, 5 + self.num_classes).to(dtype=dtype, device=device)
        preds[..., :4] = pred_boxes
        preds[..., 4:5] = pred_confs.unsqueeze(-1)
        # [N, H, W, num_classes] -> [N, H, W, 1, num_classes] -> [N, H, W, B, num_classes]
        preds[..., 5:] = pred_probs.unsqueeze(-2).expand(N, H, W, self.B, self.num_classes)

        # b_x = t_x + c_x
        # b_y = t_y + c_y
        # b_w = t_w * W
        # b_h = t_h * H
        #
        # [N, H, W, 30] -> [N, H, W, 2, 5+20]
        preds[..., 0] += x_shift
        preds[..., 1] += y_shift
        preds[..., 2] *= W
        preds[..., 3] *= H

        # Scale relative to image width/height
        preds[..., :4] *= self.stride
        # [xc, yc, w, h] -> [x1, y1, x2, y2]
        preds[..., :4] = xywh2xyxy(preds[..., :4], is_center=True)
        # [N, H, W, B, 5+num_classes] -> [B, H * W * B, 5+num_classes]
        # n_ch: [x_c, y_c, w, h, conf, class_probs]
        return preds.reshape(N, -1, 5 + self.num_classes)


class YOLOv1(nn.Module):

    def __init__(self, num_classes=20, S=7, B=2, arch='yolov1', pretrained=None):
        super(YOLOv1, self).__init__()

        self.num_classes = num_classes
        self.S = S  # 特征图大小
        self.B = B  # 每个网格单元预测的边界框数量
        self.C = num_classes  # 对象类别数量
        self.arch = arch

        if 'yolov1' == arch.lower():
            self.model = yolo.YOLOv1(num_classes=1000, S=4)
        elif 'fastyolov1' == arch.lower():
            self.model = yolo.FastYOLOv1(num_classes=1000, S=3)
        else:
            raise ValueError(f"{arch} doesn't supports")

        # self.fc = nn.Sequential(
        #     conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1,
        #                 bias=False, is_bn=False, act='relu'),
        #     conv_bn_act(1024, self.B * 5 + self.C, kernel_size=1, stride=1, padding=0,
        #                 bias=True, is_bn=True, act='identity'),
        # )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.S * self.S * (5 * self.B + self.C)),
        )

        self.yolo_layer = YOLOLayer(num_classes=self.C, S=S, B=B)

        self.__init_weights(pretrained)

    def __init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

        if pretrained is not None and pretrained != '':
            assert os.path.isfile(pretrained), pretrained
            logger.info(f'Loading pretrained {self.arch}: {pretrained}')

            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names

            self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.model.features(x)
        # x = self.features(x)

        x = self.fc(x)
        x = x.reshape(-1, self.B * 5 + self.C, self.S, self.S)

        if self.training:
            return x
        else:
            return self.yolo_layer(x)
