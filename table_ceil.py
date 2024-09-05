#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import sys, os, cv2, time, xlwt, copy
from loguru import logger as log
import numpy as np
from common.image_utils import img_process
from table_structure_recognition.table_line import table_line
from table_structure_recognition.table_detect import table_detect
from table_structure_recognition.table_build import tableBuid,to_excel
from table_structure_recognition.utils import minAreaRectbox, measure, draw_lines, draw_boxes
from common.params import args


class Table:
    def __init__(self, img, isTableDetect=False, isImgProcess=False,
                 save_path = './test', img_prefix = 'result'):
        t0 = time.time()
        self.img = copy.deepcopy(img)
        self.isImgProcess = isImgProcess
        self.save_path = save_path
        self.img_prefix = img_prefix

        self.row, self.col = 20, 20 # 距离大于该值认为是横线或者竖线
        self.alph = 30 # 两个横线或者两个竖线的任意端点的距离，如横线1的终点与横线2的起点之间的距离，若该距离小于alpha，则认为这两个点可连成一条线
        self.angle = 30 #去掉倾斜角度过大的线段
        self.interval = 20 # 表格重构过程中，若间隔距离小于interval则认为是同一条线

        self.isTableDetect = isTableDetect

        self.table_boxes_detect()  ##表格定位
        self.table_ceil()  ##表格单元格定位
        self.time_cost = time.time() - t0

    def table_boxes_detect(self):
        h, w = self.img.shape[:2]
        if self.isTableDetect:
            adBoxes, scores = table_detect(self.img)
            if len(adBoxes) <= 1: # todo
                adBoxes = [[0, 0, w, h]]
                scores = [1]
        else:
            adBoxes = [[0, 0, w, h]]
            scores = [1]

        self.adBoxes = adBoxes
        self.scores = scores


    def table_ceil(self):
        ###表格单元格
        self.newadBoxes = []
        self.newscores = []
        n = len(self.adBoxes)
        self.tableCeilBoxes = []
        self.childImgs = []
        for i in range(n):
            xmin, ymin, xmax, ymax = [int(x) for x in self.adBoxes[i]]
            xmin = max(0, xmin - 20)
            xmax = min(self.img.shape[1], xmax + 20)
            ymin = max(0, ymin - 20)
            ymax = min(self.img.shape[0], ymax + 20)
            childImg = self.img[ymin:ymax, xmin:xmax]
            # points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            # points = np.array(points, dtype = 'float32')
            # childImg = get_crop_image(self.img, points)
            if self.isImgProcess:
                childImg = cv2.cvtColor(img_process(childImg),cv2.COLOR_BGR2RGB)
            # cv2.imwrite(os.path.join(args.save_path, 'img_crop.png'), childImg)

            rowboxes, colboxes = table_line(childImg,
                                            row=self.row, col=self.col, alph=self.alph, angle=self.angle)
            tmp = np.zeros(self.img.shape[:2], dtype='uint8')
            tmp = draw_lines(tmp, rowboxes + colboxes, color=255, lineW=2)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # tmp = cv2.dilate(tmp, kernel, iterations = 1)
            if args.table_show:
                # cv2.imwrite(os.path.join(args.save_path, 'result_line.png'), tmp)
                cv2.imencode('.jpg', tmp)[1].tofile(os.path.join(args.save_path, 'result_line.png'))
            labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
            regions = measure.regionprops(labels)
            ceilboxes = minAreaRectbox(regions, False, tmp.shape[1], tmp.shape[0], filtersmall=True, adjustBox=False) # 最后一个参数改为False
            ceilboxes = np.array(ceilboxes)
            if len(ceilboxes) < 2:
                continue
            ceilboxes[:, [0, 2, 4, 6]] += xmin
            ceilboxes[:, [1, 3, 5, 7]] += ymin

            _xmin = ceilboxes[:, ::2].min()
            _xmax = ceilboxes[:, ::2].max()
            _ymin = ceilboxes[:, 1::2].min()
            _ymax = ceilboxes[:, 1::2].max()
            total_area = abs(_xmax-_xmin) * abs(_ymax-_ymin)

            c = np.array(
                [ceilboxes[:, ::2].min(axis = 1), ceilboxes[:, 1::2].min(axis = 1),
                 ceilboxes[:, ::2].max(axis = 1), ceilboxes[:, 1::2].max(axis = 1)]).T

            _ceilboxes = np.append(ceilboxes, c, axis = 1)

            area = (abs(_ceilboxes[:,8]-_ceilboxes[:,10]) * abs(_ceilboxes[:,9]-_ceilboxes[:,11])).reshape(-1,1)
            _ceilboxes = np.append(_ceilboxes, area, axis = 1)

            ceilboxes =  ceilboxes[np.where(_ceilboxes[:,-1] < 3/4*total_area)]

            self.tableCeilBoxes.append(ceilboxes.tolist())
            self.childImgs.append(childImg)
            self.newadBoxes.append(self.adBoxes[i])
            self.newscores.append(self.scores[i])

    def make_template(self, final_rboxes):
        # 做模板
        rectangle_dict = {}
        for box in range(len(final_rboxes)):
            rectangle_dict[box + 1] = np.int32(final_rboxes[box]).reshape(4,2)

        template = np.zeros(self.img.shape[:2], dtype = 'uint16')
        for r in rectangle_dict:
            cv2.fillConvexPoly(template, rectangle_dict[r], r)
        return template, rectangle_dict

    # @jit#(nopython = True)
    def table_ocr(self, final_rboxes, cor, char_out=False, ocr_result=[]):
        """use ocr and match ceil"""
        t = time.time()
        if char_out:
            ocr_result = self.character_segmentation(ocr_result)

        template, rectangle_dict = self.make_template(final_rboxes)
        content_boxes_index = list(rectangle_dict.keys())
        ceil_text = {}
        for i in content_boxes_index:
            ceil_text[i] = ""
        for m in ocr_result:
            point = [int(m["bbox"][0] + m["bbox"][2]) // 2, int(m["bbox"][1] + m["bbox"][3]) // 2]  # 中心点
            text = m["text"]
            label_ind = template[point[1]][point[0]]
            if label_ind in content_boxes_index:
                ceil_text[label_ind] += text

        for line_index in range(len(cor)):
            cor[line_index]['text'] = list(ceil_text.values())[line_index] # ocr

        print(f'match elapse: {(time.time() - t):0.2f}s')
        return cor

    def table_build(self, char_out, ocr_result):
        n = len(self.newadBoxes)
        origintableJson = []
        for i in range(n):
            final_rboxes = self.tableCeilBoxes[i]
            tablebuild = tableBuid(final_rboxes, self.interval)
            cor = tablebuild.cor
            cor = self.table_ocr(final_rboxes, cor, char_out, ocr_result)
            origintableJson.append({"box": self.newadBoxes[i], "detail": cor})

        tableJson = []
        for table in origintableJson:
            tempJson = []
            for line in table['detail']:
                if line['row'][0] < line['row'][1] and line['col'][0] < line['col'][1]:
                    tempJson.append(line)
            sorted_tempJson = sorted(tempJson, key = lambda x: (x['row'][0], x['col'][0], x['row'][1], x['col'][1]))
            tableJson.append({"box": table['box'], "detail": sorted_tempJson})
        return origintableJson, tableJson

    def __call__(self, ocr_result=[], char_out=False):
        t1 = time.time()
        origintableJson, tableJson = self.table_build(char_out, ocr_result)
        if args.table_show:
            save_path, img_prefix = self.save_path, self.img_prefix
            _tableCeilBoxes = self.tableCeilBoxes
            img = self.img
            tmp = np.zeros_like(img)

            n = len(self.newadBoxes)
            for i in range(n):
                tableCeilBoxes = _tableCeilBoxes[i]
                tmp = draw_boxes(tmp, tableCeilBoxes, color = (255, 255, 255))
                # pngP = os.path.join(args.save_path,'result_seq.png')
                pngP = os.path.join(save_path, '_'.join([img_prefix, 'seq.png']))
                # cv2.imwrite(pngP, tmp)
                cv2.imencode('.jpg', tmp)[1].tofile(pngP)

                img = draw_boxes(img, tableCeilBoxes, color = (255, 0, 0))
                # pngP = os.path.join(args.save_path, 'result_ceil.png')
                pngP = os.path.join(save_path, '_'.join([img_prefix, 'ceil.png']))
                # cv2.imwrite(pngP, img)
                cv2.imencode('.jpg', img)[1].tofile(pngP)

        if args.isToExcel:
            workbook = xlwt.Workbook(encoding = 'utf-8')
            for i in range(len(tableJson)):
                cor = tableJson[i]['detail']
                sheet = tuple([int(j) for j in tableJson[i]['box']])
                workbook = to_excel(cor, workbook = workbook, sheet = str(sheet))
            if workbook is not None and len(tableJson) != 0:
                workbook.save(os.path.join(save_path, '.'.join([img_prefix, 'xls'])))

        log.info(f"table_rec elapse: {(time.time() - t1 + self.time_cost):0.2f}s")
        return origintableJson, tableJson