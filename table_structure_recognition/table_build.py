#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :table_build.py
# @Time      :2024/09/05 11:07:24
# @Author    :Tai


import xlwt

class tableBuid:
    ##表格重建
    def __init__(self, ceilbox, interval=30):
        """
        ceilboxes:[[x1,y1,x2,y2,x3,y3,x4,y4]] #修正[[x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]]
        """
        diagBoxes =[[int(x[0]), int(x[1]), int(x[4]), int(x[5])] for x in ceilbox]

        self.diagBoxes = diagBoxes
        self.interval = interval
        self.batch()

    def batch(self):
        self.cor = []
        rowcor = self.table_line_cor(self.diagBoxes, axis='row', interval=self.interval) # 单元格在x轴方向的左边线和右边线
        colcor = self.table_line_cor(self.diagBoxes, axis='col', interval=self.interval) # 单元格在y轴方向的上边线和下边线
        cor = [{'row': line[1][:2], 'col': line[0][:2], 'box': line[0][2]} for line in zip(rowcor, colcor)]
               # if line[1][0] < line[1][1] and line[0][0] < line[0][1]] # row代表单元格的上行线和下行线， col代表单元格的左行线和右行线
        self.cor = cor

    def table_line_cor(self, lines, axis='col', interval=10):

        if axis == 'col':
            edges = [[line[1], line[3]] for line in lines]
        else:
            edges = [[line[0], line[2]] for line in lines]

        edges = sum(edges, [])
        edges = sorted(edges)

        nedges = len(edges)
        edgesMap = {}
        for i in range(nedges):
            if i == 0:
                edgesMap[edges[i]] = edges[i]
                continue
            else:
                # if edges[i] - edgesMap[edges[i - 1]] < interval: # 若间隔距离小于interval则认为是同一个
                if edges[i] - edges[i - 1] < interval:
                    edgesMap[edges[i]] = edgesMap[edges[i - 1]]
                else:
                    edgesMap[edges[i]] = edges[i]

        edgesMapList = [[key, edgesMap[key]] for key in edgesMap]
        edgesMapIndex = [line[1] for line in edgesMapList]
        edgesMapIndex = list(set(edgesMapIndex))
        edgesMapIndex = {x: ind for ind, x in enumerate(sorted(edgesMapIndex))}

        if axis == 'col':
            cor = [[edgesMapIndex[edgesMap[line[1]]], edgesMapIndex[edgesMap[line[3]]], line] for line in lines]
        else:
            cor = [[edgesMapIndex[edgesMap[line[0]]], edgesMapIndex[edgesMap[line[2]]], line] for line in lines]
        return cor

# 样式设置
def set_Style(name,size,color=0x08,borders_size=2,color_fore=0x7FFF,blod=False):
    style = xlwt.XFStyle()  # 初始化样式
    # 字体
    font = xlwt.Font()
    font.name = name
    font.height = 20 * size  # 字号
    font.bold = blod  # 加粗
    font.colour_index = color  # 默认：0x7FFF 黑色：0x08
    style.font = font
    # 居中
    alignment = xlwt.Alignment()  # 居中
    alignment.horz = xlwt.Alignment.HORZ_CENTER
    alignment.vert = xlwt.Alignment.VERT_CENTER
    style.alignment=alignment
    # 边框
    borders = xlwt.Borders()
    borders.left = xlwt.Borders.THIN
    borders.right = xlwt.Borders.THIN
    borders.top = xlwt.Borders.THIN
    borders.bottom = borders_size  # 自定义：1：细线；2：中细线；3：虚线；4：点线
    style.borders = borders
    # 背景颜色
    pattern = xlwt.Pattern()
    pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # 设置背景颜色的模式(NO_PATTERN; SOLID_PATTERN)
    pattern.pattern_fore_colour = color_fore  # 默认：无色：0x7FFF；黄色：0x0D；蓝色：0x0C
    style.pattern = pattern

    return style

def to_excel(res, workbook=None, sheet='1'):
    ##res:[{'text': '购 买 方', 'cx': 192.0, 'w': 58.0, 'h': 169.0, 'cy': 325.5, 'angle': 0.0, 'row': [0, 1], 'col': [0, 1]}]
    row = 0
    if workbook is None:
        workbook = xlwt.Workbook(encoding = 'utf-8')
    if len(res) == 0:
        worksheet = workbook.add_sheet(sheet)
        worksheet.write_merge(0, 0, 0, 0, "无数据")
    else:
        worksheet = workbook.add_sheet(sheet)
        pageRow = 0
        for line in res:
            row0, row1 = line['row']
            col0, col1 = line['col']
            text = line.get('text','')
            try:
                pageRow = max(row1 - 1, pageRow)
                worksheet.write_merge(row + row0, row + row1 - 1, col0, col1 - 1, text,
                                      set_Style('宋体',10,0x08,2,0x7FFF,blod=False))
            except:
                pass
    return workbook


if __name__=='__main__':
    pass
