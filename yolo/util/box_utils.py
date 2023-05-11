# -*- coding: utf-8 -*-

"""
@date: 2023/4/28 下午1:52
@file: box_util.py
@author: zj
@description: 
"""

import copy

import torch
from torch import Tensor


def xywh2xyxy(boxes, is_center=False):
    assert len(boxes.shape) >= 2 and boxes.shape[-1] == 4
    boxes_xxyy = copy.deepcopy(boxes)
    if is_center:
        # [x_c, y_c, w, h] -> [x1, y1, x2, y2]
        boxes_xxyy[..., 0] = (boxes[..., 0] - boxes[..., 2] / 2)
        boxes_xxyy[..., 1] = (boxes[..., 1] - boxes[..., 3] / 2)
        boxes_xxyy[..., 2] = (boxes[..., 0] + boxes[..., 2] / 2)
        boxes_xxyy[..., 3] = (boxes[..., 1] + boxes[..., 3] / 2)
    else:
        # [x1, y1, w, h] -> [x1, y1, x2, y2]
        boxes_xxyy[..., 2] = (boxes[..., 0] + boxes[..., 2])
        boxes_xxyy[..., 3] = (boxes[..., 1] + boxes[..., 3])
    return boxes_xxyy


def xyxy2xywh(boxes, is_center=False):
    assert len(boxes.shape) == 2 and boxes.shape[1] == 4
    boxes_xywh = copy.deepcopy(boxes)
    if is_center:
        # [x1, y1, x2, y2] -> [x_c, y_c, w, h]
        boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0])
        boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1])
    else:
        # [x1, y1, x2, y2] -> [x1, y1, w, h]
        boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0])
        boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1])
    return boxes_xywh


def bboxes_iou(bboxes_a: Tensor, bboxes_b: Tensor, xyxy=True) -> Tensor:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    # bboxes_a: [N_a, 4]
    # bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        # xyxy: x_top_left, y_top_left, x_bottom_right, y_bottom_right
        # 计算交集矩形的左上角坐标
        # torch.max([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        # torch.max: 双重循环
        #   第一重循环 for i in range(N_a)，遍历boxes_a, 获取边界框i，大小为[2]
        #       第二重循环　for j in range(N_b)，遍历bboxes_b，获取边界框j，大小为[2]
        #           分别比较i[0]/j[0]和i[1]/j[1]，获取得到最大的x/y
        #   遍历完成后，获取得到[N_a, N_b, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        # 计算交集矩形的右下角坐标
        # torch.min([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算bboxes_a的面积
        # x_bottom_right/y_bottom_right - x_top_left/y_top_left = w/h
        # prod([N, w/h], 1) = [N], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # prod([N_a, w/h], 1) = [N_a], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 判断符合条件的结果：x_top_left/y_top_left < x_bottom_right/y_bottom_right
    # [N_a, N_b, 2] < [N_a, N_b, 2] = [N_a, N_b, 2]
    # prod([N_a, N_b, 2], 2) = [N_a, N_b], 数值为1/0
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 首先计算交集w/h: [N_a, N_b, 2] - [N_a, N_b, 2] = [N_a, N_b, 2]
    # 然后计算交集面积：prod([N_a, N_b, 2], 2) = [N_a, N_b]
    # 然后去除不符合条件的交集面积
    # [N_a, N_b] * [N_a, N_b](数值为1/0) = [N_a, N_b]
    # 大小为[N_a, N_b]，表示bboxes_a的每个边界框与bboxes_b的每个边界框之间的IoU
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    # 计算IoU
    # 首先计算所有面积
    # area_a[:, None] + area_b - area_i =
    # [N_a, 1] + [N_b] - [N_a, N_b] = [N_a, N_b]
    # 然后交集面积除以所有面积，计算IoU
    # [N_a, N_b] / [N_a, N_b] = [N_a, N_b]
    return area_i / (area_a[:, None] + area_b - area_i)


def label2yolobox(labels):
    """
    Transform coco labels to yolo box labels
    """
    # x1/y1/w/h -> x1/y1/x2/y2
    x1 = labels[..., 0]
    y1 = labels[..., 1]
    x2 = (labels[..., 0] + labels[..., 2])
    y2 = (labels[..., 1] + labels[..., 3])

    # x1/y1/x2/y2 -> xc/yc/w/h
    labels[..., 0] = ((x1 + x2) / 2)
    labels[..., 1] = ((y1 + y2) / 2)
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # (原始高，原始宽，缩放后高，缩放后宽，ROI区域左上角x0，ROI区域左上角y0)
    h, w, nh, nw, dx, dy = info_img
    # 预测框左上角和右下角坐标
    y1, x1, y2, x2 = box
    # 计算预测框高，缩放到原始图像
    box_h = ((y2 - y1) / nh) * h
    # 计算预测框宽，缩放到原始图像
    box_w = ((x2 - x1) / nw) * w
    # 预测框左上角坐标，先将坐标系恢复到缩放后图像，然后缩放到原始图像
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    # [左上角y1，左上角x1，右下角y2，右下角x2]
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label
