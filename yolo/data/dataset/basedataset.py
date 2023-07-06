# -*- coding: utf-8 -*-

"""
@date: 2023/7/6 上午11:40
@file: basedataset.py
@author: zj
@description: 
"""

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


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
