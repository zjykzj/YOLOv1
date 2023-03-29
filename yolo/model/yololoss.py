# -*- coding: utf-8 -*-

"""
@date: 2023/3/29 下午3:15
@file: yololoss.py
@author: zj
@description: 
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def compute_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YOLOv1Loss(nn.Module):

    def __init__(self, num_classes=20, B=2, S=7, co_coord=5, co_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.B = B
        self.S = S
        self.co_coord = co_coord
        self.co_noobj = co_noobj

    def forward(self, preds: Tensor, targets: Tensor):
        """
        :param preds: [N, S, S, B*5+num_classes]
        :param targets: [N, S, S, B*5+num_classes]
        :return:
        """
        device = targets.device

        # 多少张图片
        N = preds.size()[0]
        # 获取包含目标的网格掩码。在数据预处理时已经把标注框对应网格的边界框置信度设置为1，所以只需要判断第一个边界框置信度是否为1即可
        # [N, S, S, 5*B+N_cls] -> [N, S, S] -> [N, S, S]
        coo_mask = targets[:, :, :, 4] > 0
        # 同样的，判断第一个边界框置信度是否为0即可
        # [N, S, S, 5*B+N_cls] -> [N, S, S] -> [N, S, S]
        noo_mask = targets[:, :, :, 4] == 0
        # 扩展到输入数据大小，这样，标注框对应网格的所有数据均可以保留
        # [N, S, S] -> [N, S, S, 1] -> [N, S, S, 5*B+N_cls]
        coo_mask = coo_mask.unsqueeze(-1).expand_as(targets)
        # [N, S, S] -> [N, S, S, 1] -> [N, S, S, 5*B+N_cls]
        noo_mask = noo_mask.unsqueeze(-1).expand_as(targets)

        # 提取所有符合条件的预测框数据
        # [N, S, S, 5*B+N_cls] -> [N_obj * (5*B+N_cls)] -> [N_obj, 5*b+N_cls]
        coo_pred = preds[coo_mask].view(-1, 30)
        # 提取对应的边界框信息
        # [N_obj, 5*B+N_cls] -> [N_obj * 5 * B] -> [N_obj * B, 5]
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        # 提取对应的分类信息
        # [N_obj, 5*B+N_cls] -> [N_obj, N_cls]
        class_pred = coo_pred[:, 10:]  # [x2,y2,w2,h2,c2]

        # 提取所有符合条件的真值框数据
        # [N, S, S, 5*B+N_cls] -> [N_obj * (5*B+N_cls)] -> [N_obj, 5*b+N_cls]
        coo_target = preds[coo_mask].view(-1, 30)
        # 提取对应的边界框信息
        # [N_obj, 5*B+N_cls] -> [N_obj * 5 * B] -> [N_obj * B, 5]
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        # 提取对应的分类信息
        # [N_obj, 5*B+N_cls] -> [N_obj, N_cls]
        class_target = coo_target[:, 10:]

        # compute not contain obj loss
        # 计算不包含目标的损失
        # 提取不包含目标的预测框信息
        # [N, S, S, 5*B+N_cls] -> [N_noobj * (5*B+N_cls)] -> [N_noobj, 5*B+N_cls]
        noo_pred = preds[noo_mask].view(-1, 30)
        # 提取不包含目标的真值框信息
        # [N, S, S, 5*B+N_cls] -> [N_noobj * (5*B+N_cls)] -> [N_noobj, 5*B+N_cls]
        noo_target = targets[noo_mask].view(-1, 30)
        # 创建掩码，目标是提取每个预测框的置信度
        # [N_noobj, 5*B+N_cls]
        noo_pred_mask = torch.zeros(noo_pred.shape, dtype=torch.bool)
        # 每个边界框对应的置信度掩码赋值为1
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        # 提取每个预测框的置信度
        # [N_noobj, 5*B+N_cls] -> [N_noobj * 2]
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        # 提取每个标注框的置信度（应该为0，因为不包含目标）
        # [N_noobj, 5*B+N_cls] -> [N_noobj * 2]
        noo_target_c = noo_target[noo_pred_mask]
        # 计算均值平方差损失，目标是优化不包含目标的预测框置信度为0
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')

        # compute contain obj loss
        # 计算包含目标的损失，包括
        # 1. 同一个网格中响应框的预测框损失和置信度损失，
        # 2. 同一个网格中不响应框的置信度损失，
        # 3. 包含目标的网格分类损失。
        # 创建响应掩码
        # [N_obj * B, 5]
        coo_response_mask = torch.zeros(box_target.shape, dtype=bool)
        # 创建不相应掩码
        # [N_obj * B, 5]
        coo_not_response_mask = torch.zeros(box_target.shape, dtype=bool)
        # 创建iou张量，大小设置为[N_obj * B, 5]
        box_target_iou = torch.zeros(box_target.size()).to(device)
        # 遍历每个网格（包含了B个预测框）
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            # 获取第i个网格包含的预测框数据
            # [2, 5]
            box1 = box_pred[i:i + 2]
            # 计算预测框坐标x1y1_x2y2（相对于图像宽/高的比例）
            # [2, 5]
            box1_xyxy = torch.zeros(box1.size())
            # 这里应该是有问题的？代码的目标应该是将预测框坐标转换成为x1y1_x2y2形式
            # 根据计算公式，
            # 1. 预测框前两位表示xc/yc相对于网格的比例，所以乘以1/14，表示恢复到相对于图像宽/高的比例
            # 2. u预测框后两位表示w/h相对于图像宽/高的比例
            # 所以计算结果是预测框左上角坐标和右下角坐标（相对于图像宽/高的比例）
            # 标注框的计算同理，但实际上，坐标信息的前两位表示的是预测框中心点与对应网格左上角坐标的偏移比例
            # 也就是说，
            # 想要计算预测框的左上角坐标，需要加上对应网格的左上角坐标，再乘以单个网格相对于图像的比例，
            # 才是预测框中心点对应于图像宽/高的比例
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            # 获取该网格对应标注框数据
            # [1, 5]
            box2 = box_target[i].view(-1, 5)
            # 计算标注框坐标x1y1_x2y2（相对于图像宽/高的比例）
            # [1, 5]
            box2_xyxy = torch.zeros(box2.size())
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            # 计算预测框和对应标注框的IoU
            # [2, 5], [1, 5] -> [2, 1]
            iou = compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)

            # IoU最大的预测框设置为响应，另一个设置为不响应
            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        # 1.response loss
        # 计算响应预测框
        # [N_obj * B, 5] -> [N_obj * 5] -> [N_obj, 5]
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        # [N_obj * B, 5] -> [N_obj * 5] -> [N_obj, 5]
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        # [N_obj * B, 5] -> [N_obj * 5] -> [N_obj, 5]
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        # 计算均值平方差损失，目的是优化响应预测框和标注框之间的IoU（target=1）
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        # 计算坐标损失，
        # 1. 对于中心点坐标，直接计算均值平方差损失
        # 2. 对于长宽，先进行平方根，平滑小目标和大目标对于相同偏差距离带来的不同影响，然后计算均值平方差损失
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + \
                   F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]),
                              reduction='sum')
        # 2.not response loss
        # 计算未响应目标的预测框损失
        # [N_obj * B, 5] -> [N_obj * (B-1) * 5] -> [N_obj * (B-1), 5]
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        # [N_obj * B, 5] -> [N_obj * (B-1) * 5] -> [N_obj * (B-1), 5]
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        # 设置未响应预测框的置信度为0
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # I believe this bug is simply a typo
        # 计算置信度损失
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 3.class loss
        # 计算分类损失
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.co_coord * loc_loss +
                contain_loss +
                self.co_noobj * not_contain_loss + self.co_noobj * nooobj_loss +
                class_loss) / N


if __name__ == '__main__':
    m = YOLOv1Loss()
    print(m)

    a = torch.randn(1, 7, 7, 30)
    b = torch.zeros(1, 7, 7, 30)
    b[:, 2, 3] = torch.abs(torch.randn(30))
    b[:, 2, 3, 4] = 1
    b[:, 2, 3, 9] = 1
    b[:, 2, 3, 17] = 1
    loss = m(a, b)
    print(loss)

    m = m.cuda()
    loss = m(a.cuda(), b.cuda())
    print(loss)
