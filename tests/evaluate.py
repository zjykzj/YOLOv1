# -*- coding: utf-8 -*-

"""
@date: 2023/4/27 上午9:35
@file: vocevaluate.py
@author: zj
@description:
"""
import os
import glob

import numpy as np
from tqdm import tqdm

from yolo.data.evaluate.vocevaluator import VOCEvaluator
from yolo.data.dataset.vocdataset import VOCDataset


def test_voc_evaluate():
    print("=> Test VOC Evaluate")

    root = '../datasets/voc'
    VOCdevkit_dir = os.path.join(root, 'VOCdevkit')
    year = 2007
    split = 'test'
    val_evaluator = VOCEvaluator(VOCDataset.classes, VOCdevkit_dir, year=year, split=split)
    print(val_evaluator)

    all_boxes_dict = dict()

    result_dir = 'tests/results/VOC2007/Main/'
    print(f"=> Read results from {result_dir}\n")
    res_path_list = glob.glob(os.path.join(result_dir, '*.txt'))
    for res_path in tqdm(res_path_list):
        res_name = os.path.splitext(res_path)[0]
        class_name = res_name.split('_')[-1]

        box_list = list()
        with open(res_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue

                tmp_list = line.strip().split(' ')
                assert len(tmp_list) == 6, line
                image_name, score, x1, y1, x2, y2 = tmp_list[:6]
                box_list.append([image_name, float(score), float(x1), float(y1), float(x2), float(y2)])
        assert class_name not in all_boxes_dict.keys()
        all_boxes_dict[class_name] = box_list

    print(f"all_boxes_dict.keys():\n{sorted(all_boxes_dict.keys())}")
    print(f"VOCDataset.classes:\n{sorted(VOCDataset.classes)}")
    val_evaluator.all_boxes_dict = all_boxes_dict
    ap50, ap50_95 = val_evaluator.result()
    print('ap50:', ap50)
    print('ap50_95:', ap50_95)
    assert np.allclose(ap50_95, 0.752207)


if __name__ == '__main__':
    test_voc_evaluate()
