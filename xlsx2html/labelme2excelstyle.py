#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :labelme2excelstyle.py
# @Time      :2024/05/11 10:39:10
# @Author    :Tai


import os
from collections import defaultdict
from loguru import logger as log


def rowcol2excelstyle(table):
    box = table['box']
    detail = table['detail']
    rows = defaultdict(list)
    style = {'style': {'height': '13.0pt', 'border-collapse': 'collapse', 'border-right-style':
             'solid', 'border-right-width': '1px', 'border-right-color': '#000000',
               'border-left-style': 'solid', 'border-left-width': '1px',
               'border-left-color': '#000000', 'border-top-style': 'solid',
               'border-top-width': '1px', 'border-top-color': '#000000',
               'border-bottom-style': 'solid', 'border-bottom-width': '2px',
               'border-bottom-color': '#000000', 'text-align': 'center', 'background-color': None,
               'font-size': '10.0px', 'color': '#000000'}}
    for d in detail:
        column = d['col'][0]
        row = d['row'][0]
        value = d['text']

        row_num = d['row'][1] - row
        if row_num > 1:
            rowspan = row_num
        elif row_num == 1:
            rowspan = None
        else:
            log.error(f'rowspan == {rowspan}, < 1')

        col_num = d['col'][1] - column
        if col_num > 1:
            colspan = col_num
        elif col_num == 1:
            colspan = None
        else:
            log.error(f'colspan == {colspan}, < 1')

        attrs = {'colspan': colspan, 'rowspan': rowspan}
        temp_dict = {'column': column, 'row': row, 'value': value, 'formatted_value': value, 'attrs': attrs}
        temp_dict.update(style)
        rows[d['row'][0]].append(temp_dict)

    rows = list(rows.values())
    for row in rows:
        row.sort(key = lambda x: x['column'])
    data = {'rows': rows, 'cols': [], 'images': defaultdict(list)}
    data.update({'imagePath': str(box)})
    return data



if __name__ == "__main__":
    json_file = r'./test/train_img/1.json'
    data = rowcol2excelstyle(json_file)

    from excelstyle2html import re_html
    html_file = os.path.splitext(json_file)[0] + '.HTML'
    re_html(data, html_file)

