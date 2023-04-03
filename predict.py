# -*- coding: utf-8 -*-

"""
@date: 2023/4/3 上午11:18
@file: predict.py
@author: zj
@description: 
"""

import cv2
import PIL

import numpy as np

import torch
import torchvision.transforms as transforms

from yolo.model.yolov1 import YOLOv1

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def nms(bboxes, scores, threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0].item()
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def main():
    S = 7
    B = 2
    C = 20
    model = YOLOv1(num_classes=C, S=S, B=B)
    # ckpt_path = './best.pth.tar'
    ckpt_path = './checkpoint.pth.tar'
    print(f"Load {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names
    model.load_state_dict(state_dict, strict=True)

    device = torch.device('cuda:0')
    model = model.to(device)

    # image_name = 'dog'
    # img_path = './dog.jpg'
    # image_name = '000283'
    # img_path = './000283.jpg'
    # image_name = '000011'
    # img_path = './000011.jpg'
    #
    # val
    image_name = '2008_000864'
    img_path = './2008_000864.jpg'
    image = PIL.Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])
    ])
    data = transform(image).unsqueeze(0).to(device)

    outputs = model(data)[0].detach().cpu()
    print(outputs.shape)

    conf_thre = 0.1

    pred_box = outputs[:, :, :B * 4]
    # [S, S, B]
    pred_conf = outputs[:, :, B * 4:B * 5]
    # [S, S, B]
    mask1 = pred_conf > conf_thre
    # [S, S, B]
    # mask2 = pred_conf == pred_conf.max()
    # [S, S, B]
    # conf_mask = (mask1 + mask2).gt(0)
    conf_mask = mask1

    boxes = list()
    probs = list()
    cls_indexes = list()

    cell_size = 1. / S
    for i in range(S):
        for j in range(S):
            for bi in range(B):
                if conf_mask[i, j, bi] == 1:
                    # [xc, yc, w, h]
                    box = pred_box[i, j, bi * 4:(bi + 1) * 4]
                    xc = (box[0] + i) * cell_size
                    yc = (box[1] + j) * cell_size
                    w = box[2]
                    h = box[3]

                    box[0] = xc - 0.5 * w
                    box[1] = yc - 0.5 * h
                    box[2] = xc + 0.5 * w
                    box[3] = yc + 0.5 * h

                    pred_conf = outputs[i, j, bi]
                    max_prob, max_idx = torch.max(outputs[i, j, B * 5:], 0)
                    if float(max_prob * pred_conf) > conf_thre:
                        boxes.append(box)
                        probs.append(pred_conf * max_prob)
                        cls_indexes.append(max_idx)

    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexes = torch.zeros(1)
    else:
        boxes = torch.stack(boxes)  # (n,4)
        probs = torch.stack(probs)  # (n,)
        cls_indexes = torch.stack(cls_indexes)  # (n,)
    keep = nms(boxes, probs)
    boxes, cls_indexes, probs = boxes[keep], cls_indexes[keep], probs[keep]

    image = np.array(image)[..., ::-1].astype(np.uint8)
    height, width = image.shape[:2]

    result = list()
    for i, box in enumerate(boxes):
        x1 = int(box[0] * width)
        x2 = int(box[2] * width)
        y1 = int(box[1] * height)
        y2 = int(box[3] * height)
        cls_index = cls_indexes[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])

    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv2.imwrite('result.jpg', image)


if __name__ == '__main__':
    main()
