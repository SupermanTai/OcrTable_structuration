#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tableJson2html.py
# @Time      :2024/05/11 10:39:15
# @Author    :Tai


import os, warnings
warnings.filterwarnings('ignore')

from xlsx2html.excelstyle2html import re_html
from xlsx2html.labelme2excelstyle import rowcol2excelstyle
from xlsx2html.html2tablestructurer import html2tablestructurer
# from loguru import logger as log
from common.params import args

def tableJson2html(tableJson, save_path, img_prefix):
    html_list = []
    len_ = len(tableJson)
    if len_ == 1:
        data = rowcol2excelstyle(tableJson[0])
        html_file = os.path.join(save_path, f"{img_prefix}.HTML")
        html = re_html(data)
        html = html2tablestructurer(html)
        if args.table_show:
            with open(html_file, 'w', encoding = 'utf8') as f:
                f.write(html)
        html_list.append(html)
    else:
        for idx, table in enumerate(tableJson):
            data = rowcol2excelstyle(table)
            html_file = os.path.join(save_path, f"{img_prefix}_{idx}.HTML")
            html = re_html(data)
            html = html2tablestructurer(html)
            if args.table_show:
                with open(html_file, 'w', encoding = 'utf8') as f:
                    f.write(html)
            html_list.append(html)
    return html_list



# if __name__ == "__main__":
#     # json_file = r'test/9.json'
#     # labelme2html(json_file)
#     img_folder = r'test/train_img'
#     result = tableJson2html(img_folder, mode = 'train')