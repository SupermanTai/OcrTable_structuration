#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :html2tablestructurer.py
# @Time      :2024/05/11 10:39:05
# @Author    :Tai


import copy, os
import re, json
from bs4 import BeautifulSoup
import ast

def html2tablestructurer(html_str):
    html = BeautifulSoup(html_str, 'lxml')
    html_table_attrs = 'border=6 width=500px bgcolor=#f2f2f2 cellspacing=0 cellpadding=5 align=center'
    html_table_attrs = html_table_attrs.split(' ')
    html.table.attrs = {i.split('=')[0]: i.split('=')[1] for i in html_table_attrs}

    # filename = html.find('title').text

    tbody = html.find('tbody')
    if tbody is None:
        tbody = html.find('table')
    tr_list = tbody.find_all('tr')
    for tr in tr_list:
        td_list = tr.find_all('td')
        for td in td_list:
            # td_text = td.text
            # if td_text != '':
            #     bbox = ast.literal_eval(td_text)
            # else:
            #     bbox = []
            # cell = {"bbox": bbox, 'tokens': ''}
            # dic['html']['cells'].append(cell)
            td_attrs = copy.deepcopy(td.attrs)
            td.attrs = {}
            try:
                rowspan = td_attrs['rowspan']
                td.attrs['rowspan'] = rowspan
            except:
                pass
            try:
                colspan = td_attrs['colspan']
                td.attrs['colspan'] = colspan
            except:
                pass

    # pattern = re.compile(r'<[^>]+>', re.S)
    # str_tbody = str(tbody)
    # tokens = pattern.findall(str_tbody)
    # tokens_new = []
    # for i in tokens:
    #     temp = i.split(' ')
    #     for j, t in enumerate(temp):
    #         if j == 0:
    #             temp[j] = t
    #         else:
    #             temp[j] = ' ' + t
    #     tokens_new.extend(temp)
    #
    # tokens_new2 = []
    # for i in tokens_new:
    #     if 'span' not in i:
    #         tokens_new2.append(i)
    #     else:
    #         temp = [i[:-1], i[-1]]
    #         tokens_new2.extend(temp)
    # dic['html']['structure']['tokens'] = tokens_new2

    simple_html_str = '<html><body><table "border-collapse: collapse; width: 95%;" border="1" cellspacing="0" cellpadding="10">' + \
                        str(tbody) + '</table></body></html>'
    # dic['html']['gt'] = simple_html_str
    # dic_json = json.dumps(dic, ensure_ascii = False)
    return simple_html_str



if __name__ == "__main__":
    html_str = '''
        <html><body><table><tr><td colspan=\"2\">篇目</td><td>哲思与情怀</td></tr><tr><td rowspan=\"2\">《庄子》二则</td><td>北冥有鱼</td><td></td></tr><tr><td>庄子与惠子游于濠梁之上</td><td></td></tr><tr><td rowspan=\"2\">《礼记》二则</td><td>虽有佳肴</td><td></td></tr><tr><td>大道之行</td><td></td></tr></table></body></html>
    '''
    res = html2tablestructurer(html_str)