#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy, time
# import torch.nn.functional as F
import math

from table_structure_recognition.utils import get_table_line, adjust_lines, final_adjust_lines
import numpy as np
import cv2
from common.params import args

from table_structure_recognition.inference import Inference

model = Inference(is_slide = args.is_slide)

def table_line(image, row=50, col=30, alph=15, angle=50):
    # image = Image.fromarray(img)
    pred = model(image, is_resize=args.is_resize)
    pred = np.uint8(pred)
    hpred = copy.deepcopy(pred)  # 横线
    vpred = copy.deepcopy(pred)  # 竖线
    whereh = np.where(hpred == 1)
    wherev = np.where(vpred == 2)
    hpred[wherev] = 0
    vpred[whereh] = 0

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # vpred = cv2.dilate(vpred, kernel, iterations = 1)
    # hpred = cv2.dilate(hpred, kernel, iterations = 1)

    hpred_origion = copy.deepcopy(hpred)
    vpred_origion = copy.deepcopy(vpred)
    # 膨胀算法的色块大小
    h, w = pred.shape
    hors_k = int(math.sqrt(w) * 1.2)
    vert_k = int(math.sqrt(h) * 1.2)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    vpred = cv2.morphologyEx(vpred, cv2.MORPH_CLOSE, vkernel, iterations = 1)  # 先膨胀后腐蚀的过程
    hpred = cv2.morphologyEx(hpred, cv2.MORPH_CLOSE, hkernel, iterations = 1)

    if args.table_show:
        import matplotlib.pyplot as plt
        plt.imshow(vpred+hpred)
        plt.show()

    colboxes = get_table_line(vpred, axis=1, lineW=col) # 竖线
    rowboxes = get_table_line(hpred, axis=0, lineW=row) # 横线

    start = time.time()
    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph = alph, angle=angle)
    print(f'adjust_lines elapse: {(time.time() - start):0.2f}s')

    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    start = time.time()
    rowboxes, colboxes = final_adjust_lines(rowboxes, colboxes)
    print(f'final_adjust_lines elapse: {(time.time() - start):0.2f}s')

    # rowboxes, colboxes = filter_lines(rowboxes, colboxes, angle = 2)

    # rowboxes = overlapping_filter(rowboxes, 1, separation = 5)
    # colboxes = overlapping_filter(colboxes, 0, separation = 5)
    return rowboxes, colboxes


