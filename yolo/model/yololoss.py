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

    class_target = torch.zeros((N, H * W, C)).to(dtype=dtype, device=device)
    class_mask = torch.zeros((N, H * W, 1)).to(dtype=dtype, device=device)

    return iou_target, iou_mask, box_target, box_mask, class_target, class_mask


class YOLOv1Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

    def build_target(self, outputs, targets):
        N, H, W, n_ch = outputs.shape
        assert n_ch == (self.B * 5 + self.C)
        assert H == W == self.S

        dtype = outputs.dtype
        device = outputs.device

        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = \
            build_mask(N, H, W, B=self.B, C=self.C, dtype=dtype, device=device)
        iou_mask *= self.lambda_noobj

        # grid coordinate
        # [W] -> [H, W, B]
        x_shift = torch.broadcast_to(torch.arange(W).reshape(1, W, 1),
                                     (H, W, self.B)).to(dtype=dtype, device=device)
        # [H] -> [H, 1] -> [H, W, B]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(H, 1, 1),
                                     (H, W, self.B)).to(dtype=dtype, device=device)

        all_pred_boxes = outputs[..., :8].reshape(N, H, W, 2, 4)
        all_pred_boxes = torch.sigmoid(all_pred_boxes)
        all_pred_boxes[..., 0] += x_shift.expand(N, H, W, self.B)
        all_pred_boxes[..., 1] += y_shift.expand(N, H, W, self.B)

        # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
        gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

        for ni in range(N):
            num_obj = gt_num_objs[ni]
            if num_obj == 0:
                iou_mask[ni, ...] = 0
                continue

            gt_boxes = targets[ni][:num_obj][..., :4]
            gt_cls_ids = targets[ni][:num_obj][..., 4]

            pred_boxes = all_pred_boxes[ni][..., :4].reshape(-1, 4)

            # [H*W*B, 4] x [num_obj, 4] -> [H*W*B, num_obj] -> [H*W, B, num_obj]
            ious = bboxes_iou(pred_boxes, gt_boxes, xyxy=False).reshape(H * W, self.B, num_obj)
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)

            for oi in range(num_obj):
                gt_box = gt_boxes[oi]
                gt_class = gt_cls_ids[oi]

                cell_idx_x, cell_idx_y = torch.floor(gt_box[:2])
                cell_idx = cell_idx_y * W + cell_idx_x
                cell_idx = cell_idx.long()

                class_target[ni, cell_idx, int(gt_class)] = 1
                class_mask[ni, cell_idx, :] = 1

                # which predictor
                ious_in_cell = ious[cell_idx, :, oi]
                argmax_idx = torch.argmax(ious_in_cell)

                target_delta = gt_box
                target_delta[0] -= cell_idx_x
                target_delta[1] -= cell_idx_y

                box_target[ni, cell_idx, argmax_idx, :] = target_delta
                box_mask[ni, cell_idx, argmax_idx, :] = 1

                iou_target[ni, cell_idx, argmax_idx] = max_iou[cell_idx, argmax_idx, :]
                iou_mask[ni, cell_idx, argmax_idx] = self.lambda_obj

        return iou_target.reshape(-1), \
            iou_mask.reshape(-1), \
            box_target.reshape(-1, 4), \
            box_mask.reshape(-1, 1), \
            class_target.reshape(-1, self.C), \
            class_mask.reshape(-1, 1)

    def forward(self, outputs, targets):
        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = \
            self.build_target(outputs.detach().clone(), targets)

        N, H, W, n_ch = outputs.shape
        assert H == W == self.S
        assert n_ch == (5 * self.B + self.C)

        outputs = torch.sigmoid(outputs)
        # [N*H*W*B, 4]
        pred_boxes = outputs[..., :(4 * self.B)].reshape(N, H, W, 2, 4).reshape(-1, 4)
        # [N*H*W*B]
        pred_confs = outputs[..., (4 * self.B):(5 * self.B)].reshape(-1)
        # [N*H*W, C]
        pred_probs = outputs[..., (5 * self.B):].reshape(-1, self.C)

        # iou loss
        pred_confs = pred_confs * iou_mask
        iou_target = iou_target * iou_mask
        loss_conf = F.mse_loss(pred_confs, iou_target, reduction='sum')

        # box loss
        pred_boxes = pred_boxes * box_mask
        box_target = box_target * box_mask

        loss_xy = F.mse_loss(pred_boxes[..., :2], box_target[..., :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(pred_boxes[..., 2:4]), torch.sqrt(box_target[..., 2:4]), reduction='sum')

        # class loss
        pred_probs = pred_probs * class_mask
        class_target = class_target * class_mask

        loss_class = F.mse_loss(pred_probs, class_target, reduction='sum')

        # total
        loss = (loss_xy + loss_wh) * self.lambda_coord + loss_conf + loss_class * self.lambda_class
        return loss
