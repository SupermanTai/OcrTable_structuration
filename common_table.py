# -*- coding: utf-8 -*-

import warnings, os, time, cv2
warnings.filterwarnings("ignore")
from loguru import logger as log
if not os.path.exists('./log'):
    os.makedirs('./log')
# log.remove()  # 删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
log.add(os.path.join('./log', f'{time.strftime("%Y_%m_%d")}.log'), retention="10 days")

from loading import OCR, args, text_sys
import table_ceil
import numpy as np
from common.timeit import time_it
from xlsx2html.tableJson2html import tableJson2html


@time_it
def main(image_file, stream=False, save_path = './test', img_prefix = 'result'):
    if stream:
        img = cv2.imdecode(np.frombuffer(image_file, dtype = np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(np.fromfile(image_file, dtype = np.uint8), cv2.IMREAD_COLOR)

    ocr = OCR(text_sys, img, cls = False)
    ocr_result = ocr()

    tableRec = table_ceil.Table(img, isTableDetect=False, save_path = save_path, img_prefix = img_prefix)
    _, tableJson = tableRec(ocr_result)
    html_list = tableJson2html(tableJson, save_path, img_prefix)
    assert len(tableJson) == len(html_list)
    [i.update({'html': html_list[idx]}) for idx, i in enumerate(tableJson)]
    return tableJson


if __name__ == "__main__":
    import os

    # image_file = r''
    image_file = r'141.png'
    save_path, img_prefix = os.path.dirname(image_file), os.path.basename(image_file).split('.')[0]
    tableJson = main(image_file, save_path = save_path, img_prefix=img_prefix)
    print(tableJson)
