#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ocr_utils.py
# @Time      :2024/09/05 11:06:07
# @Author    :Tai

import math

from PIL import Image
from scipy.ndimage import filters, interpolation
from numpy import amin, amax
import numpy as np
import base64
from io import BytesIO
import cv2
from loguru import logger as log
from common.params import args
from shapely.geometry import Polygon
import pyclipper

def resize_im(im, scale, max_scale=None, return_f=False):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    if return_f:
        if f >=1:
            return cv2.resize(im, (0, 0), fx=f, fy=f, interpolation = cv2.INTER_CUBIC), f
        else:
            return cv2.resize(im, (0, 0), fx=f, fy=f, interpolation = cv2.INTER_AREA), f
    else:
        if f >=1:
            return cv2.resize(im, (0, 0), fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
        else:
            return cv2.resize(im, (0, 0), fx = f, fy = f)

def estimate_skew_angle(raw, angleRange=[-15, 15]):
    """
    估计图像文字偏转角度,
    angleRange:角度估计区间
    """
    raw = Image.fromarray(raw)
    raw = np.array(raw.convert('L'))
    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw - amin(raw)
    image = image / amax(image)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0 / 0.5)
    # w,h = image.shape[1],image.shape[0]
    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    flat = np.clip(image[:h, :w] - m[:h, :w] + 1, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    flat = amax(flat) - flat
    flat -= amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    angles = range(angleRange[0], angleRange[1])
    estimates = []
    for a in angles:
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))

    _, a = max(estimates)
    return a

def eval_angle(img, degree):
    """
    估计图片文字的偏移角度
    """
    im = Image.fromarray(img)
    # degree = estimate_skew_angle(np.array(im.convert('L')), angleRange=angleRange)
    im = im.rotate(degree, center=(im.size[0] / 2, im.size[1] / 2), expand=1, fillcolor=(255, 255, 255))
    img = np.array(im)
    return img


from math import fabs, sin, cos, radians
def get_img_rot_broa(img, degree=90, cal_reM=True, borderValue=(255, 255, 255), min_angle=0.0):
    # if abs(degree) < min_angle or abs(degree) > 90 - min_angle:
    #     return img, None, None

    height, width = img.shape[:2]
    height_new = int(width * fabs(sin(radians(degree))) +
                     height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) +
                    width * fabs(cos(radians(degree))))
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    # inv_mat_rotation = np.linalg.pinv(mat_rotation)
    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=borderValue)
    if cal_reM:
        inv_mat_rotation = np.copy(mat_rotation) * -1
        inv_mat_rotation[0,0] *= -1
        inv_mat_rotation[1,1] *= -1
    else:
        inv_mat_rotation = None
    return img_rotated, mat_rotation, inv_mat_rotation


from numpy import cos, sin
def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new

def recalrotateposition(x, y, angle, img, img_new):
    if angle:
        (image_h, image_w) = img.shape[:2]
        cx, cy = image_w / 2, image_h / 2
        new_x, new_y = rotate(x, y, angle * np.pi / 180, cx, cy)
        diffx = (img_new.shape[1] - img.shape[1]) / 2
        diffy = (img_new.shape[0] - img.shape[0]) / 2
        new_x, new_y = new_x + diffx, new_y + diffy
    else:
        new_x, new_y = x, y
    return new_x, new_y


def cal_center(box_lst):
    '''
    box: [121.0, 25.0, 206.0, 25.0, 206.0, 53.0, 121.0, 53.0]
    '''
    center_lst = []
    for box in box_lst:
        x1, y1 = box[0], box[1]  # left-upper
        x3, y3 = box[4], box[5]  # right-bottom
        w = x3 - x1
        h = y3 - y1
        center_x = x1 + w // 2
        center_y = y1 + h // 2
        center_lst.append([center_x, center_y])
    return center_lst

def string_to_arrimg(img_str, log_flag=False):
    '''
    读取base64字符，保存图像
    '''
    try:
        if isinstance(img_str, str):
            img_str = img_str.encode('utf-8')
        img_data = base64.b64decode(img_str)

        img_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 不能转为RGB，否则会出错
        if log_flag:
            log.info('image shape is:%s'%str(img.shape))
        return img
    except:
        return None

def arrimg_to_string(img):
    '''
    图像数据转化为base64
    '''
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format='png')
    img = buffer.getvalue()
    img_str = base64.b64encode(img).decode('utf8')
    return img_str

def arrimg2string(img):
    img = cv2.imencode('.jpg', img)[1]
    base64_data = base64.b64encode(img).decode('utf8')
    return base64_data

def imagefile_to_string(filename):
   with open(filename,"rb") as f:#转为二进制格式
       img_str = base64.b64encode(f.read()).decode('utf8')#使用base64进行加密
       #print(base64_data)
   return img_str

def get_sub_img(img, scale):
    '''
    抠图
    '''
    m, n, _ = img.shape
    x_min = int(min(scale[:, 0]))
    x_max = int(max(scale[:, 0]))
    y_min = int(min(scale[:, 1]))
    y_max = int(max(scale[:, 1]))

    sub_img = img[max(0, y_min): min(m, y_max), max(0, x_min): min(n, x_max):]
    return sub_img


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    # dst_img_height, dst_img_width = dst_img.shape[0:2]
    # if dst_img_height * 1.0 / dst_img_width >= 1.5:
    #     dst_img = np.rot90(dst_img)
    return dst_img


## 图片旋转
def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])

    # # 计算图像的新边界尺寸
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # #nH = h

    # # 调整旋转矩阵
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


## 获取图片旋转角度
def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def fourxy2twoxy(quadrangle):
    quadrangle = np.array(quadrangle)
    pot_lf = float(min(quadrangle[:, 0]))
    pot_tp = float(min(quadrangle[:, 1]))
    pot_rt = float(max(quadrangle[:, 0]))
    pot_bm = float(max(quadrangle[:, 1]))
    bbox = [pot_lf, pot_tp, pot_rt, pot_bm]
    return bbox

def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    box = box.reshape(1, -1).squeeze().tolist()
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def text_area(img, bbox_list, left_ratio=0.05, right_ratio=0.05,top_ratio=0.01,bottom_ratio=0.05):
    bbox_list = np.array(bbox_list)
    top_left = int(bbox_list[:,0].min()), int(bbox_list[:,1].min())
    bottm_right = int(bbox_list[:,2].max()), int(bbox_list[:,3].max())

    left = int(img.shape[1]*left_ratio)
    right = int(img.shape[1]*right_ratio)
    top = int(img.shape[0]*top_ratio)
    bottom = int(img.shape[0]*bottom_ratio)

    x_min = max(0, top_left[0]-left)
    x_max = min(img.shape[1], bottm_right[0]+right)
    y_min = max(0, top_left[1]-top)
    y_max = min(img.shape[0], bottm_right[1]+bottom)
    # if (y_max - y_min) / img.shape[0] < 0.8:
    #     crop_img = img[y_min:y_max, :]
    # else:
    #     crop_img = img
    # if (x_max - x_min) / img.shape[1] < 0.8:
    #     crop_img = crop_img[:, x_min:x_max]
    crop_img = img[y_min:y_max, x_min:x_max]
    if args.is_visualize:
        cv2.imwrite(r'test/adjust.jpg', crop_img)

    return crop_img, x_min, y_min

def cal_angle(dt_boxes, std_max: float = 10.):
    angles, heights, widths = [], [], []
    for box in dt_boxes:
        rect = cv2.minAreaRect(np.int32(box).reshape(-1,1,2))
        _, (w, h), alpha = rect
        a, w_, h_, cx, cy = solve(box)
        if w_ < h_:  # todo 2022-11-23 过滤竖直文本
            continue
        # if alpha == 0 or abs(alpha) == 90:
        if math.isclose(alpha, 0, abs_tol = 1) or math.isclose(alpha, 90, abs_tol = 1):
            continue
        widths.append(w)
        heights.append(h)
        angles.append(alpha)

    if len(angles) <= 1:
        return 0

    # max_widths = np.max(widths)
    # max_heights = np.max(heights)
    # if max_widths < max_heights:
    #     index_list = [i for i in range(len(heights)) if heights[i] >= 1 / 3 * max_heights]
    # else:
    #     index_list = [i for i in range(len(widths)) if widths[i] >= 1 / 3 * max_widths]
    # heights = np.array(heights)[index_list]
    # widths = np.array(widths)[index_list]
    # angles = np.array(angles)[index_list]

    # if len(angles) <= 1:
    #     return 0

    if len(angles) > 1:
        sorted_index_list = sorted(range(len(angles)), key = lambda k: angles[k])
        angles = np.array(angles)[sorted_index_list]
        widths = np.array(widths)[sorted_index_list]
        heights = np.array(heights)[sorted_index_list]
        angles = angles[1:-1]
        widths = widths[1:-1]
        heights = heights[1:-1]

    if len(angles) <= 1:
        return 0

    if np.std(angles) > std_max:
        # Edge case with angles of both 0 and 90°, or multi_oriented docs
        angle = 0
    else:
        angle = -np.mean(angles)
        # Determine rotation direction (clockwise/counterclockwise)
        # Angle coverage: [-90°, +90°], half of the quadrant
        if np.sum(widths) < np.sum(heights):  # CounterClockwise
            if cv2.__version__ >= '4.5':
                angle = 90 + angle
            else:
                angle = -(90 - angle)
    return angle



def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key = lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])

def uncliped_bbox(quadrangle, unclip_ratio, img_height, img_width):
    # quadrangle = convert_coord(bbox)
    quadrangle = unclip(quadrangle, unclip_ratio).reshape(-1, 1, 2)
    quadrangle, sside = get_mini_boxes(quadrangle)
    quadrangle = np.array(quadrangle)
    quadrangle[:, 0] = np.clip(np.round(quadrangle[:, 0]), 0, img_width)
    quadrangle[:, 1] = np.clip(np.round(quadrangle[:, 1]), 0, img_height)
    return quadrangle

def convert_coord(xyxy):
    """
    Convert two points format to four points format.
    :param xyxy:
    :return:
    """
    new_bbox = np.zeros([4,2], dtype=np.float32)
    new_bbox[0,0], new_bbox[0,1] = xyxy[0], xyxy[1]
    new_bbox[1,0], new_bbox[1,1] = xyxy[2], xyxy[1]
    new_bbox[2,0], new_bbox[2,1] = xyxy[2], xyxy[3]
    new_bbox[3,0], new_bbox[3,1] = xyxy[0], xyxy[3]
    return new_bbox