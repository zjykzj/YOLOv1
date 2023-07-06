# -*- coding: utf-8 -*-

"""
@date: 2023/7/6 上午11:40
@file: basedataset.py
@author: zj
@description: 
"""
import os

from typing import List, Union
from numpy import ndarray

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from ..target import Target


class BaseDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        pass

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def __len__(self):
        pass

    def set_img_size(self, img_size):
        self.target_size = img_size

    def get_img_size(self):
        return self.target_size

    def _build_target(self, labels: ndarray, img_info: List, img_id: Union[int, str]):
        assert isinstance(labels, ndarray)
        target = torch.zeros((self.max_det_nums, 5))
        if len(labels) > 0:
            for i, label in enumerate(labels[:self.max_det_nums]):
                target[i, :] = torch.from_numpy(label)

        if self.train:
            return target
        else:
            image_name = os.path.splitext(os.path.basename(img_id))[0]
            target = Target(target, img_info, image_name)
        return target
