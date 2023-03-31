# -*- coding: utf-8 -*-

"""
@date: 2023/3/30 下午4:59
@file: loss_zj.py
@author: zj
@description: 
"""

import torch
from torch import nn
import torch.nn.functional as F


def bbox_iou(boxes1, boxes2):
    assert len(boxes1.shape) == 2, boxes1
    assert len(boxes2.shape) == 2, boxes2
    # boxes1 shape: (N, 4)
    # boxes2 shape: (M, 4)
    N = boxes1.size(0)
    M = boxes2.size(0)

    lt = torch.max(boxes1[:, :2].unsqueeze(1).expand(N, M, 2),
                   boxes2[:, :2].unsqueeze(0).expand(N, M, 2))
    rb = torch.min(boxes1[:, 2:].unsqueeze(1).expand(N, M, 2),
                   boxes2[:, 2:].unsqueeze(0).expand(N, M, 2))

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # [N, M]
    iou = inter / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter)

    return iou


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

    def forward(self, predictions, targets):
        # predictions shape: (batch_size, S, S, (B*5+C))
        # targets shape: (batch_size, S, S, (B*5+C))
        assert predictions.shape == targets.shape

        device = predictions.device

        N = len(predictions)
        assert self.S == predictions.shape[2] and self.S == predictions.shape[2]
        assert (self.B * 5 + self.C) == predictions.shape[-1]

        # extract predicted objectness scores, predicted bounding boxes, and predicted class probabilities
        # [N, S, S, B]
        pred_conf = predictions[..., self.B * 4:self.B * 5]
        # [N, S, S, B*4] -> [N, S, S, B, 4]
        pred_boxes = predictions[..., :self.B * 4].reshape(N, self.S, self.S, self.B, 4)
        # [N, S, S, C]
        pred_class = predictions[..., self.B * 5:]
        # extract target objectness scores, target bounding boxes, and target class probabilities
        target_conf = targets[..., self.B * 4:self.B * 5]
        target_boxes = targets[..., :self.B * 4].reshape(N, self.S, self.S, self.B, 4)
        target_class = targets[..., self.B * 5:]

        # Compute the binary mask for the presence of objects in each grid cell.
        # [N, S, S, B]
        obj_mask = target_conf.clone()
        obj_mask[obj_mask > 0] = 1
        obj_mask[obj_mask <= 0] = 0

        # Compute the binary mask for the absence of objects in each grid cell.
        # [N, S, S, B]
        noobj_mask = 1 - obj_mask

        # 计算损失如下：
        # 1. 包含标注框的网格中响应框的坐标损失
        # 2. 包含标注框的网格中响应框的置信度损失
        # 3. 包含标注框的网格中不响应框的置信度损失
        # 4. 不包含标注框的网格对应预测框的置信度损失
        # 5. 包含标注框的网格分类损失
        loss_for_coord = 0.
        loss_for_response = 0.
        loss_for_noresponse = 0.
        loss_for_noobj = 0.
        loss_for_cls = 0.

        cell_size = 1. / self.S

        # 遍历图像
        for ni in range(N):
            # 遍历网格
            for i in range(self.S):
                for j in range(self.S):
                    # 判断该网格是否包含标注框
                    if obj_mask[ni, i, j, 0] == 1:
                        # 计算IoU，IoU最大的设置为响应框，负责预测标注框
                        # [B, 4]: [xc_offset, yc_offset, box_w, box_h]
                        pred_grid_boxes = pred_boxes[ni, i, j]
                        pred_xc = (pred_grid_boxes[:, 0] + i) * cell_size
                        pred_yc = (pred_grid_boxes[:, 1] + j) * cell_size
                        pred_w = pred_grid_boxes[:, 2]
                        pred_h = pred_grid_boxes[:, 3]

                        pred_x1 = pred_xc - 0.5 * pred_w
                        pred_y1 = pred_yc - 0.5 * pred_h
                        pred_x2 = pred_xc + 0.5 * pred_w
                        pred_y2 = pred_yc + 0.5 * pred_h

                        # [1, 4]
                        target_grid_boxes = target_boxes[ni, i, j, 0].unsqueeze(0)
                        target_xc = (target_grid_boxes[:, 0] + i) * cell_size
                        target_yc = (target_grid_boxes[:, 1] + j) * cell_size
                        target_w = target_grid_boxes[:, 2]
                        target_h = target_grid_boxes[:, 3]

                        target_x1 = target_xc - 0.5 * target_w
                        target_y1 = target_yc - 0.5 * target_h
                        target_x2 = target_xc + 0.5 * target_w
                        target_y2 = target_yc + 0.5 * target_h

                        # iou([B, 4], [1, 4]) -> [B, 1]
                        ious = bbox_iou(
                            torch.concat((pred_x1, pred_y1, pred_x2, pred_y2)).reshape(self.B, 4),
                            torch.concat((target_x1, target_y1, target_x2, target_y2)).reshape(1, 4)
                        )
                        max_iou_id = torch.argmax(ious).detach().cpu().item()

                        for bi in range(self.B):
                            if bi == max_iou_id:
                                # 计算响应框置信度损失
                                loss_for_response += (pred_conf[ni, i, j, max_iou_id] - 1.) ** 2

                                # 计算坐标损失
                                loss_for_coord += F.mse_loss(pred_boxes[ni, i, j, max_iou_id],
                                                             target_boxes[ni, i, j, max_iou_id], reduction='sum')
                            else:
                                # 计算不响应框置信度损失
                                loss_for_noresponse += (pred_conf[ni, i, j, bi] - 0.) ** 2

                        # 计算网格分类损失
                        loss_for_cls += F.mse_loss(pred_class[ni, i, j], target_class[ni, i, j], reduction='sum')
                    else:
                        # 不包含标注框，仅计算置信度损失
                        loss_for_noobj += F.mse_loss(pred_conf[ni, i, j],
                                                     torch.zeros(target_conf[ni, i, j].shape).to(device),
                                                     reduction='sum')

        loss = self.lambda_coord * loss_for_coord + \
               self.lambda_obj * loss_for_response + \
               self.lambda_noobj * loss_for_noresponse + \
               self.lambda_noobj * loss_for_noobj + \
               self.lambda_class * loss_for_cls
        return loss / N


if __name__ == '__main__':
    m = YOLOv1Loss()
    print(m)

    torch.manual_seed(32)

    B = 2
    S = 7
    N_cls = 20
    a = torch.randn(1, S, S, 5 * B + N_cls)
    b = torch.zeros(1, S, S, 5 * B + N_cls)
    b[:, 2, 3] = torch.abs(torch.randn(5 * B + N_cls))
    for bi in range(B):
        b[:, 2, 3, 5 * bi + 4] = 1
    b[:, 2, 3, 5 * B + 8] = 1
    loss = m(a, b)
    print(loss)

    m = m.cuda()
    loss = m(a.cuda(), b.cuda())
    print(loss)

    # a = torch.abs(torch.randn(3, 4)) * 100
    # # b = torch.abs(torch.randn(4, 4)) * 100
    # b = torch.abs(torch.randn(1, 4)) * 100
    # c = bbox_iou(a, b)
    # print(c.shape)
    # print(c)
    #
    # print(torch.argmax(c, dim=1))
