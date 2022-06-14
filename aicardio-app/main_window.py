import os
import cv2
import sys
import pydicom
import time
import numpy as np

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtMultimedia import *
from PySide6.QtCharts import *
from functools import partial

from src.main_ui import *
from src.dthreads import *
from src.config import *
from src.database import MyDB
db = MyDB()


class MainWindow(QMainWindow, Ui_MainWindow):
    dicomThread = {
        2: None,
        3: None,
        4: None,
    }
    s2c = [0]*6
    s3c = [0]*6
    s4c = [0]*6
    gls = 0
    vidSender = None
    chamber = None
    timer = QTimer()
    newBullEye = True

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setStyleSheet("background-color: dark;")

        self.setupUi(self)
        self._setupFolderView()
        self._setupButtons()
        self._setupDisplay()
        self._setupDonut()

    def _setupFolderView(self):
        self.fileModel = QFileSystemModel()
        self.fileModel.setRootPath(QDir.rootPath())
        self.fileModel.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)

        self.folderView.setModel(self.fileModel)
        self.folderView.setAnimated(False)
        self.folderView.setIndentation(20)
        self.folderView.setSortingEnabled(True)
        self.folderView.hideColumn(1)
        self.folderView.hideColumn(2)
        self.folderView.hideColumn(3)
        self.folderView.setHeaderHidden(True)
        self.folderView.setRootIndex(self.fileModel.index(QDir.currentPath()))
        self.folderView.doubleClicked.connect(self._changeFile)

    def _setupButtons(self):
        self.processButton.clicked.connect(self._processVideo)
        self.changeFolderButton.clicked.connect(self._changeFolder)

    def _changeFolder(self):
         dlg = QFileDialog()
         dlg.setFileMode(QFileDialog.Directory)
         dlg.setViewMode(QFileDialog.Detail)
         if dlg.exec():
             fpath = dlg.selectedFiles()[0]
             print(f'Folder changed to : {fpath}')
             self.folderView.setRootIndex(self.fileModel.index(fpath))

    def _setupDisplay(self):
        for i in [2, 3, 4]:
            self.chamberView[i].mousePressEvent = partial(self._changeChamber,
                                                          chamberWidget=self.chamberView[i],
                                                          chamber=i)
        self.efGraph.mousePressEvent = partial(self._changeChamber,
                                               chamberWidget=self.frameInfo,
                                               chamber=0)

        # setup periodic events
        self.timer.timeout.connect(self._updateGraph)
        self.timer.timeout.connect(self._updateBullEye)
        self.timer.timeout.connect(self._updateEF)
        self.timer.start(100) # 1000/fps

    def _setupDonut(self):
        def genSlice(value):
            slc = QPieSlice(f'{value:.1f}', 100+value/1e6)
            slc.setLabelVisible(True)
            slc.setLabelColor(Qt.black)
            slc.setLabelPosition(QPieSlice.LabelInsideHorizontal)

            val = min(1.0, (value / -30.0) + 0.3)
            color = QColor(Qt.red)
            color.setRedF(val)
            slc.setBrush(color)
            return slc
        def genDonut(data, holeSize, pieSize, sAngle=-360, eAngle=-360):
            donut = QPieSeries()
            for val in data:
                donut.append(genSlice(val))
            donut.setHoleSize(holeSize)
            donut.setPieSize(pieSize)
            donut.setPieStartAngle(360+sAngle)
            donut.setPieEndAngle(720+eAngle)
            return donut

        # using 16-segment model
        # https://core.ac.uk/reader/53744593?utm_source=linkout
        # https://asecho.org/wp-content/uploads/2018/08/WFTF-Chamber-Quantification-Summary-Doc-Final-July-18.pdf
        self.newBullEye = False
        s2c, s3c, s4c = self.s2c, self.s3c, self.s4c

        # 1, 6, 5, 4, 3, 2
        basalData = [s2c[-1], s4c[-1], s3c[0], s2c[0], s4c[0], s3c[-1]]
        # 7, 12, 11, 10, 9, 8
        midData = [s2c[-2], s4c[-2], s3c[1], s2c[1], s4c[1], s3c[-2]]
        # 13, 16, 15, 14
        apicalData = [s2c[-3], (s4c[-3]+s3c[2])/2.0, s2c[2], (s4c[2]+s3c[-3])/2.0]

        self.basalDonut = genDonut(basalData, 0.7, 1.05, -30, -30)
        self.midDonut = genDonut(midData, 0.35, 0.7, -30, -30)
        self.apicalDonut = genDonut(apicalData, 0.0, 0.35, -45, -45)

        for donut in [self.basalDonut, self.midDonut, self.apicalDonut]:
            self.bullEye.chart().addSeries(donut)

    def _updateBullEye(self):
        # periodically check and update bull eye
        for thread in self.dicomThread.values():
            if thread is None: return
            if not thread.isResultLoaded: return
        if not self.newBullEye: return

        self.s2c = self.dicomThread[2].sls
        self.s3c = self.dicomThread[3].sls
        self.s4c = self.dicomThread[4].sls

        self.gls2c = self.dicomThread[2].gls
        self.gls3c = self.dicomThread[3].gls
        self.gls4c = self.dicomThread[4].gls
        self.gls = (self.gls2c+self.gls3c+self.gls4c) / 3.0
        self._setupDonut()

    def _updateEF(self):
        efs, glss = [], []
        for chamber in [2, 3, 4]:
            thread = self.dicomThread.get(chamber, None)
            if (not thread) or (not thread.isResultLoaded):
                self.efLabel[chamber].setText('')
                self.glsLabel[chamber].setText('')
                continue

            ef = thread.ef
            gls = thread.gls
            self.efLabel[chamber].setText(f'{ef:.1f}%')
            self.glsLabel[chamber].setText(f'{gls:.1f}')
            efs.append(ef)
            glss.append(gls)

        if len(efs) == 3:
            # show mean values
            self.efLabel[0].setText(f'{np.mean(efs):.1f}%')
            self.glsLabel[0].setText(f'{np.mean(glss):.1f}')

            a2c = self.dicomThread[2]
            a2c_ed_idx = a2c.ed_idx
            a2c_es_idx = a2c.es_idx
            a2c_ed_dat = a2c.frame_datas[a2c_ed_idx]
            a2c_es_dat = a2c.frame_datas[a2c_es_idx]

            a4c = self.dicomThread[4]
            a4c_ed_idx = a4c.ed_idx
            a4c_es_idx = a4c.es_idx
            a4c_ed_dat = a4c.frame_datas[a4c_ed_idx]
            a4c_es_dat = a4c.frame_datas[a4c_es_idx]

            a2c_ed_rads = a2c_ed_dat['radius_list']
            a4c_ed_rads = a4c_ed_dat['radius_list']
            a2c_ed_h = a2c_ed_dat['avg_height']
            a4c_ed_h = a4c_ed_dat['avg_height']
            ed_vol = self._compute_volume(a2c_ed_rads, a4c_ed_rads, a2c_ed_h, a4c_ed_h)

            a2c_es_rads = a2c_es_dat['radius_list']
            a4c_es_rads = a4c_es_dat['radius_list']
            a2c_es_h = a2c_es_dat['avg_height']
            a4c_es_h = a4c_es_dat['avg_height']
            es_vol = self._compute_volume(a2c_es_rads, a4c_es_rads, a2c_es_h, a4c_es_h)

            self.efLabel[0].setText(f'{((ed_vol-es_vol) / ed_vol)*100:.2f}%')

    def _compute_volume(self, a2c_rads, a4c_rads, a2c_h, a4c_h):
        h = (a2c_h+a4c_h) / 2.0
        vol = 0
        for rad2, rad4 in zip(a2c_rads, a4c_rads):
            vol += np.pi * rad2 * rad4 * h
        vol += np.pi * h * a2c_rads[-1] * a4c_rads[-1] # add 1 more cylinder
        return vol

    def _updateGraph(self):
        # periodically update the ef graph
        thread = self.dicomThread.get(self.chamber, None)
        if thread:
            efList = thread.efList
            self.efGraph.plot(list(range(len(efList))), efList, symbol='+', clear=True)
        else:
            self.efGraph.clear()

        if self.chamber == 0:
            self.frameInfo.setCurrentIndex(self.bullEyeIdx)
        else:
            self.frameInfo.setCurrentIndex(self.bullEyeIdx)
            #self.frameInfo.setCurrentIndex(self.efGraphIdx)

    def _changeChamber(self, event, chamberWidget=None, chamber=None):
        def reset(widget):
            widget.setLineWidth(1)
            widget.setFrameStyle(QFrame.Box | QFrame.Plain)

        [reset(self.chamberView[i]) for i in [2, 3, 4]]; reset(self.frameInfo)
        chamberWidget.setFrameStyle(QFrame.Box | QFrame.Raised)
        chamberWidget.setLineWidth(3)
        self.chamber = chamber

        print('Chamber changed to {}'.format(chamber))

    def _updateFrame(self, data, widget):
        if data.maskedFrame is not None:
            img = data.maskedFrame
        else:
            img = data.rawFrame

        img = img.scaled(FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = QPixmap.fromImage(img)
        widget.setPixmap(pixmap)

    def _capVid(self, fpath, chamber):
        # capture new video when file changed
        newVidThread = DicomThread(fpath, chamber=chamber)
        newVidThread.change_pixmap_signal.connect(partial(self._updateFrame, widget=self.chamberView[chamber]))
        if self.dicomThread[chamber]: self.dicomThread[chamber].stop()
        self.dicomThread[chamber] = newVidThread
        self.dicomThread[chamber].start()
        self._changeStatus(f'Playing file: {fpath} at chamber {self.chamber}')

        fname = os.path.basename(fpath)
        self.chamberLabel[chamber].setText(f'{chamber}C - {fname}')

    @Slot()
    def _processVideo(self):
        if not self.dicomThread.get(self.chamber, None): return
        newVidSender = VideoSender(self.dicomThread[self.chamber].fpath)
        newVidSender.startSignal.connect(self._hideButton)
        newVidSender.msgSignal.connect(self._changeStatus)
        if self.vidSender is not None: self.vidSender.stop()
        self.vidSender = newVidSender
        self.vidSender.start()

    @Slot(bool)
    def _hideButton(self, isHidden):
        self.processButton.setEnabled(not isHidden)

    @Slot(str)
    def _changeStatus(self, msg):
        self.statusBar.showMessage(msg)

    @Slot()
    def _changeFile(self):
        fpath = self.fileModel.filePath(self.folderView.currentIndex())
        if os.path.isdir(fpath): return
        if not pydicom.misc.is_dicom(fpath):
            self._changeStatus('File is not a DICOM, please select a DICOM file')
            return
        if not self.chamber:
            self._changeStatus('Please select a chamber to play file')
            return
        self.newBullEye = True
        print('File changed to {}'.format(fpath))
        self._capVid(fpath, self.chamber)

    def closeEvent(self, event):
        if self.vidSender: self.vidSender.stop()
        for thread in self.dicomThread.values():
            if thread: thread.stop()
        db.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
