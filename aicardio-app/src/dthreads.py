import os
import cv2
import sys
import time
import copy
import pydicom
import numpy as np

from dataclasses import dataclass
from pydicom.pixel_data_handlers.util import convert_color_space
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtMultimedia import *
from PySide6.QtCharts import *

from src.database import MyDB
from src.utils import *


db = MyDB()


def cv2qt(cvImg):
    h, w, ch = cvImg.shape
    rgbImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    qImg = QImage(rgbImg.data, w, h, ch*w, QImage.Format_RGB888)
    return qImg


@dataclass()
class DisplayData:
    blackImg = cv2qt(np.zeros((400, 600, 3), np.uint8))
    rawFrame: QImage = blackImg.copy()
    maskedFrame: QImage = None


class VideoSender(QThread):
    startSignal = Signal(bool)
    msgSignal = Signal(str)

    def __init__(self, fpath):
        super().__init__()
        self.fpath = fpath

    def run(self):
        self.startSignal.emit(True)
        self.msgSignal.emit(f'Sending file: {self.fpath}')
        if db.sendFile(self.fpath):
            self.msgSignal.emit('Send done! Waiting for results.')
        else:
            self.msgSignal.emit('Send failed!')
        self.startSignal.emit(False)

    def stop(self):
        self.wait()


class DicomThread(QThread):
    change_pixmap_signal = Signal(DisplayData)
    _run_flag = True

    frames = []
    masks = []
    efList = []
    data = {}
    isResultLoaded = False
    timer = MyTimer()

    def __init__(self, fpath=None, chamber=None):
        super().__init__()
        self.fpath = fpath
        self.chamber = chamber

    def _loadFile(self):
        dataset = pydicom.dcmread(self.fpath)
        frames = dataset.pixel_array
        frames = convert_color_space(frames, 'YBR_FULL', 'RGB')
        if len(frames.shape) == 3 and frames.shape[-1] != 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        self.frames = frames
        self.qFrames = [cv2qt(frame) for frame in self.frames]

    def _draw(self, frames, contours):
        #masks = [draw_contour(fr, ct) for fr, ct in zip(self.frames, contours)]
        masks = []
        frame_datas = []
        for frame, contour in zip(frames, contours):
            mask = draw_contour(frame, contour)
            avg_height, radius, lines, peak, mid_point = biplane(contour)
            for (p1, p2) in lines:
                mask = draw_line(mask, p1, p2)
            mask = draw_line(mask, peak, mid_point)

            dat = {}
            dat['radius_list'] = radius
            dat['lines'] = lines
            dat['avg_height'] = avg_height
            dat['peak'] = peak
            dat['mid_point'] = mid_point

            masks.append(mask)
            frame_datas.append(dat)
        return masks, frame_datas

    def _loadResults(self):
        if self.isResultLoaded: return
        data = db.getResults(self.fpath)

        if data.get('status', '') == 'done':
            contours = data['results']['pivot_sequence']
            self.masks, self.frame_datas = self._draw(self.frames, contours)
            self.qMasks = [cv2qt(mask) for mask in self.masks]
            self.efList = list(data['results']['volumes'])
            print('New draw updated')

            self.data = data['results']
            self.sls = data['results']['SLS']
            self.gls = data['results']['GLS']
            self.ef = abs(data['results']['ef'])
            self.ed_idx = data['results']['max_idx']
            self.es_idx = data['results']['min_idx']

            self.isResultLoaded = True

    def run(self):
        self._loadFile()
        self._loadResults()
        idx = -1
        nFrame = len(self.frames)
        while self._run_flag:
            self.timer.tik()
            idx = (idx + 1) % nFrame
            qImg = self.qFrames[idx]

            qMask = None
            if len(self.masks) > idx:
                qMask = self.qMasks[idx]

            dat = DisplayData(rawFrame=qImg, maskedFrame=qMask)
            self.change_pixmap_signal.emit(dat)

            self._loadResults()
            self.timer.tok(fps=30)

    def stop(self):
        self._run_flag = False
        self.wait()
