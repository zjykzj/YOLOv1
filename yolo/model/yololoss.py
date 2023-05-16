# -*- coding: utf-8 -*-

"""
@date: 2023/5/11 下午5:49
@file: yololoss.py
@author: zj
@description: 
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from yolo.util.box_utils import bboxes_iou


def make_deltas(box1: Tensor, box2: Tensor) -> Tensor:
    """
    t_x = b_x - c_x
    t_y = b_y - c_y
    t_w = b_w
    t_h = b_h

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """
    assert len(box1.shape) == len(box2.shape) == 2
    # [N, 4] -> [N]
    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2]
    t_h = box2[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def build_mask(N, H, W, B=2, C=20, dtype=torch.float, device=torch.device('cpu')):
    iou_target = torch.zeros((N, H * W, B)).to(dtype=dtype, device=device)
    iou_mask = torch.ones(N, H * W, B).to(dtype=dtype, device=device)

    box_target = torch.zeros((N, H * W, B, 4)).to(dtype=dtype, device=device)
    box_mask = torch.zeros((N, H * W, B, 1)).to(dtype=dtype, device=device)

    # class_target = torch.zeros((N, H * W, C)).to(dtype=dtype, device=device)
    class_target = torch.zeros((N, H * W, 1)).to(dtype=dtype, device=device)
    class_mask = torch.zeros((N, H * W, 1)).to(dtype=dtype, device=device)

    return iou_target, iou_mask, box_target, box_mask, class_target, class_mask


class YOLOv1Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20, ignore_thresh=0.5,
                 lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.ignore_thresh = ignore_thresh

        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

    def build_target(self, outputs, targets):
        N, n_ch, H, W = outputs.shape
        assert n_ch == (self.B * 5 + self.C)
        assert H == W == self.S

        dtype = outputs.dtype
        device = outputs.device

        # [N, n_ch, H, W] -> [N, H, W, n_ch]
        outputs = outputs.permute(0, 2, 3, 1)

        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = \
            build_mask(N, H, W, B=self.B, C=self.C, dtype=dtype, device=device)

        # grid coordinate
        # [W] -> [1, 1, W, 1] -> [N, H, W, B]
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, 1, W, 1),
                                     (N, H, W, self.B)).to(dtype=dtype, device=device)
        # [H] -> [1, H, 1, 1] -> [N, H, W, B]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(1, H, 1, 1),
                                     (N, H, W, self.B)).to(dtype=dtype, device=device)

        all_pred_boxes = outputs[..., :(self.B * 4)].reshape(N, H, W, self.B, 4)
        all_pred_boxes = torch.sigmoid(all_pred_boxes)
        all_pred_boxes[..., 0] += x_shift
        all_pred_boxes[..., 1] += y_shift
        all_pred_boxes[..., 2] *= W
        all_pred_boxes[..., 3] *= H

        # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
        gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

        for ni in range(N):
            num_obj = gt_num_objs[ni]
            if num_obj == 0:
                iou_mask[ni, ...] = 0
                continue

            gt_cls_ids = targets[ni][:num_obj][..., 4]
            gt_boxes = targets[ni][:num_obj][..., :4]
            gt_boxes[..., 0::2] *= W
            gt_boxes[..., 1::2] *= H

            pred_boxes = all_pred_boxes[ni][..., :4].reshape(-1, 4)

            # [H*W*B, 4] x [num_obj, 4] -> [H*W*B, num_obj] -> [H*W, B, num_obj]
            ious = bboxes_iou(pred_boxes, gt_boxes, xyxy=False).reshape(H * W, self.B, num_obj)
            # Maximum IoU corresponding to each prediction box
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)

            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            # [H*W, B, 1] -> [H*W*B] -> [n_pos]
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                iou_mask[ni][max_iou.squeeze() >= self.ignore_thresh] = 0

            for oi in range(num_obj):
                gt_box = gt_boxes[oi]
                gt_class = gt_cls_ids[oi]

                cell_idx_x, cell_idx_y = torch.floor(gt_box[:2])
                cell_idx = cell_idx_y * W + cell_idx_x
                cell_idx = cell_idx.long()

                # class_target[ni, cell_idx, int(gt_class)] = 1
                class_target[ni, cell_idx, :] = int(gt_class)
                class_mask[ni, cell_idx, :] = 1

                # which predictor
                ious_in_cell = ious[cell_idx, :, oi]
                argmax_idx = torch.argmax(ious_in_cell)

                target_delta = gt_box
                target_delta[0] -= cell_idx_x
                target_delta[1] -= cell_idx_y
                target_delta[2] /= W
                target_delta[3] /= H

                box_target[ni, cell_idx, argmax_idx, :] = target_delta
                box_mask[ni, cell_idx, argmax_idx, :] = 1

                iou_target[ni, cell_idx, argmax_idx] = max_iou[cell_idx, argmax_idx, :]
                iou_mask[ni, cell_idx, argmax_idx] = 2

        return iou_target.reshape(-1), \
            iou_mask.reshape(-1), \
            box_target.reshape(-1, 4), \
            box_mask.reshape(-1), \
            class_target.reshape(-1).long(), \
            class_mask.reshape(-1)

    def forward(self, outputs, targets):
        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = \
            self.build_target(outputs.detach().clone(), targets)

        N, n_ch, H, W = outputs.shape
        assert H == W == self.S
        assert n_ch == (5 * self.B + self.C)

        outputs = outputs.permute(0, 2, 3, 1)
        # Compres xywh/conf/prob to [0, 1]
        # outputs = torch.sigmoid(outputs)
        outputs[..., :(self.B * 5)] = torch.sigmoid(outputs[..., :(self.B * 5)])

        # [N*H*W*B, 4]
        pred_boxes = outputs[..., :(self.B * 4)].reshape(N, H, W, self.B, 4).reshape(-1, 4)
        # [N*H*W*B]
        pred_confs = outputs[..., (self.B * 4):(self.B * 5)].reshape(-1)
        # [N*H*W, C]
        pred_probs = outputs[..., (self.B * 5):].reshape(-1, self.C)

        # iou loss
        pred_confs_obj = pred_confs[iou_mask == 2]
        iou_target_obj = iou_target[iou_mask == 2]
        loss_obj = F.mse_loss(pred_confs_obj, iou_target_obj, reduction='sum')

        pred_confs_noobj = pred_confs[iou_mask == 1]
        iou_target_noobj = iou_target[iou_mask == 1]
        loss_noobj = F.mse_loss(pred_confs_noobj, iou_target_noobj, reduction='sum')

        # box loss
        pred_boxes = pred_boxes[box_mask > 0]
        box_target = box_target[box_mask > 0]

        loss_xy = F.mse_loss(pred_boxes[..., :2], box_target[..., :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(pred_boxes[..., 2:4]), torch.sqrt(box_target[..., 2:4]), reduction='sum')

        # class loss
        pred_probs = pred_probs[class_mask > 0]
        class_target = class_target[class_mask > 0]
        # loss_class = F.mse_loss(pred_probs, class_target, reduction='sum')
        loss_class = F.cross_entropy(pred_probs, class_target, reduction='sum')

        # total
        loss = (loss_xy + loss_wh) * self.lambda_coord + \
               loss_obj * self.lambda_obj + loss_noobj * self.lambda_noobj + \
               loss_class * self.lambda_class
        return loss
