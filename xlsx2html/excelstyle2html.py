#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :excelstyle2html.py
# @Time      :2024/05/11 10:38:59
# @Author    :Tai


import re
import io
from xlsx2html.core import render_data_to_html

def excelstyle2html(data, output=None,
              append_headers=(lambda dumb1, dumb2: True),
              append_lineno=(lambda dumb1, dumb2: True),
              ):

    html = render_data_to_html(data, append_headers, append_lineno)

    # if not output:
    #     output = io.StringIO()
    # if isinstance(output, str):
    #     output = open(output, 'w', encoding = 'utf8') # todo
    # output.write(html)
    # return output
    return html

def re_html(data):
    html = excelstyle2html(data)
    # with open(html_file, encoding = 'utf8') as f:
    #     html = f.read()
    t1 = re.findall(r'<body>(.*?)</body>', html, re.S)[0]
    t1 = str(t1).replace('font-size: 12.0px','font-size: 16.0px')\
        .replace('"border-collapse: collapse"','"border-collapse: collapse; width: 95%;"').\
        replace('font-size: 10.5px','font-size: 16.0px').\
        replace('border-left: none;','border-left-style: solid;border-left-width: 1px;').\
        replace('border-right: none;','border-right-style: solid;border-right-width: 1px;').\
        replace('border-top: none;','border-top-style: solid;border-top-width: 1px;').\
        replace('font-size: 11.0px','font-size: 16.0px')

    html = '''
        <!DOCTYPE html>
        <html lang="zh">
        <head>
            <meta charset="UTF-8">
            <title>%s</title>
        </head>
        <body>
            %s
        </body>
        </html>
        '''
    html = html %(data['imagePath'], t1)
    # with open(html_file, 'w', encoding = 'utf8') as f:
    #     f.write(html)
    return html

