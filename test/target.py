# -*- coding: utf-8 -*-

"""
@date: 2023/3/30 下午3:52
@file: target.py
@author: zj
@description: 
"""

import numpy as np

# Define grid size, number of bounding boxes and number of classes
S = 3
B = 2
num_classes = 5

# Define the ground truth bounding boxes for the image
# [x1, y1, x2, y2]
gt_boxes = np.array([[0.3, 0.4, 0.7, 0.8], [0.6, 0.2, 0.9, 0.5]])

# Define the ground truth class labels for the bounding boxes
gt_labels = np.array([2, 3])

# Initialize the target tensor with all zeros
target = np.zeros((S, S, 5 * B + num_classes))

# Calculate the size of each grid cell
cell_size = 1 / float(S)

# Encode the ground truth information in the target tensor
for i in range(gt_boxes.shape[0]):
    box = gt_boxes[i]
    x_center, y_center, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
    x_idx, y_idx = int(x_center // cell_size), int(y_center // cell_size)
    x_offset, y_offset = x_center / cell_size - x_idx, y_center / cell_size - y_idx
    class_onehot = np.zeros(num_classes)
    class_onehot[gt_labels[i]] = 1
    for j in range(B):
        if target[y_idx, x_idx, j * 5] == 0:
            target[y_idx, x_idx, j * 5:(j + 1) * 5] = np.array([x_offset, y_offset, w, h, 1])
            target[y_idx, x_idx, 5 * B + gt_labels[i]] = 1
            break

# The target tensor is of shape (S, S, 5 * B + num_classes)
print(target.shape)
print(target)