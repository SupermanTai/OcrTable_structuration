#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :timeit.py
# @Time      :2024/05/11 10:37:51
# @Author    :Tai


import time
from loguru import logger as log

class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        log.info("------------------ Time Start ----------------------")
        self.st = time.time()

    def end(self, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st)
        else:
            self.time = (self.et - self.st)

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 2)


class Timer(Times):
    def __init__(self):
        super(Timer, self).__init__()

    def info(self):
        total_time = self.value()
        total_time = round(total_time, 2)
        log.info("total_time(s): {}".format(total_time))
        log.info("------------------ Time End ----------------------")

def time_it(func):
    def inner(*args, **kwargs):
        log.info("------------------ Inference Start ----------------------")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total_time = round(end-start, 2)
        log.info("total_time: {}s".format(total_time))
        log.info("------------------ Inference End ------------------------")
        return result
    return inner