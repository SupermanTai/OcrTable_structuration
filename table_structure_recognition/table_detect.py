#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :table_detect.py
# @Time      :2024/09/05 11:07:30
# @Author    :Tai

import cv2, os
import numpy as np
from common.params import args
from table_structure_recognition.predict_layout import LayoutPredictor
import paddle
from common.ocr_utils import uncliped_bbox, fourxy2twoxy, convert_coord

layout_predictor = LayoutPredictor(args)


def table_detect(img):
    layout_res, elapse = layout_predictor(img)
    paddle.device.cuda.empty_cache()
    boxes = []
    confidences = []
    for region in layout_res:
        if region['label'] == 'table':
            x1, y1, x2, y2 = region['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # roi_img = self.img[y1:y2, x1:x2, :]
            # cv2.imwrite('./test/img_crop.jpg', roi_img)
            adBoxes = [x1, y1, x2, y2]
            quadrangle = convert_coord(adBoxes)
            quadrangle = uncliped_bbox(quadrangle, unclip_ratio = 0.3, img_height = img.shape[0],
                                       img_width = img.shape[1])
            adBoxes = fourxy2twoxy(quadrangle)
            adBoxes = [int(i) for i in adBoxes]

            scores = region['score']
            boxes.append(adBoxes)
            confidences.append(scores)
    # return boxes, confidences
    if len(boxes) > 0:
        order_index = np.array(boxes)[:, -1].argsort()
        boxes = np.array(boxes)[order_index].tolist()
        confidences = np.array(confidences)[order_index].tolist()
    return boxes, confidences