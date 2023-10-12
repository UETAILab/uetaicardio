import pyqtgraph as pg
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtMultimedia import *
from PySide6.QtCharts import *
from PySide6.QtQuick import *


from src.config import *


class Ui_MainWindow(object):
    def _addChamberLabel(self, parent, text):
        layout = QVBoxLayout(parent)
        label = QLabel()
        label.setObjectName(u"{text}")
        label.setStyleSheet("QLabel { color : green; }")
        label.setFont(QFont('Arial', 15))
        label.setText(text)
        layout.addWidget(label, alignment=Qt.AlignTop|Qt.AlignHCenter)
        return label

    def _createChamberView(self, name):
        chamberView = QLabel()
        chamberView.setScaledContents(True)
        chamberView.setObjectName(name)
        chamberView.setFrameStyle(QFrame.Box | QFrame.Plain)
        chamberView.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        return chamberView

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(0, 0))
        self.horizontalLayout = QHBoxLayout(self.centralwidget)

        self.dataWidget = QWidget(self.centralwidget)
        self.dataLayout = QGridLayout(self.dataWidget)
        self.dataLayout.setObjectName(u"dataLayout")
        self.dataLayout.setContentsMargins(0, 0, 0, 0)

        self.frame2C = self._createChamberView(u"frame2C")
        self.label2C = self._addChamberLabel(self.frame2C, u"2C")
        self.dataLayout.addWidget(self.frame2C, 0, 0, 1, 1, alignment=Qt.AlignCenter)

        self.frame3C = self._createChamberView(u"frame3C")
        self.label3C = self._addChamberLabel(self.frame3C, u"3C")
        self.dataLayout.addWidget(self.frame3C, 0, 1, 1, 1, alignment=Qt.AlignCenter)

        self.frame4C = self._createChamberView(u"frame4C")
        self.label4C = self._addChamberLabel(self.frame4C, u"4C")
        self.dataLayout.addWidget(self.frame4C, 1, 0, 1, 1, alignment=Qt.AlignCenter)

        self.chamberView = {
            2: self.frame2C,
            3: self.frame3C,
            4: self.frame4C,
        }
        self.chamberLabel = {
            2: self.label2C,
            3: self.label3C,
            4: self.label4C,
        }

        self.frameInfo = QStackedWidget()
        self.frameInfo.setFrameStyle(QFrame.Box | QFrame.Plain)

        self.bullEye = QChartView()
        self.bullEye.setRenderHint(QPainter.Antialiasing)
        self.chart = self.bullEye.chart()
        #self.chart.setTitle("BullEye")
        self.chart.setTheme(QChart.ChartThemeDark)
        self.chart.legend().setVisible(False)
        self.chart.setBackgroundVisible(False)

        def genGLSView(text, parentLayout):
            layout = QHBoxLayout()
            label = QLabel()
            label.setText(text)

            valLabel = QLabel()
            valLabel.setFixedWidth(40)
            layout.addWidget(label, alignment=Qt.AlignLeft)
            layout.addWidget(valLabel, alignment=Qt.AlignLeft)
            parentLayout.addLayout(layout)
            return layout, valLabel

        self.resLayout = QVBoxLayout()
        self.resLayout.setContentsMargins(*[3]*4)
        self.glsLabel = {}
        self.a2cGLSLayout, self.glsLabel[2] = genGLSView('A2C-GLS:', self.resLayout)
        self.a3cGLSLayout, self.glsLabel[3] = genGLSView('A3C-GLS:', self.resLayout)
        self.a4cGLSLayout, self.glsLabel[4] = genGLSView('A4C-GLS:', self.resLayout)
        self.avgGLSLayout, self.glsLabel[0] = genGLSView('meanGLS:', self.resLayout)
        self.resLayout.addSpacing(10)
        self.efLabel = {}
        self.a2cEFLayout, self.efLabel[2] = genGLSView('A2C-EF:', self.resLayout)
        self.a3cEFLayout, self.efLabel[3] = genGLSView('A3C-EF:', self.resLayout)
        self.a4cEFLayout, self.efLabel[4] = genGLSView('A4C-EF:', self.resLayout)
        self.avgEFLayout, self.efLabel[0] = genGLSView('biplaneEF:', self.resLayout)

        self.resView = QGroupBox(self.bullEye)
        self.resView.setLayout(self.resLayout)
        self.resView.setFlat(True)

        #self.efGraph = pg.PlotWidget()
        #self.efGraph.setObjectName(u"efGraph")
        #self.efGraph.setTitle("LV Volumes", color="g", size="15pt")
        #styles = {'color':'w', 'font-size':'10px'}
        #self.efGraph.setLabel('left', 'Volume', **styles)
        #self.efGraph.setLabel('bottom', 'Frame index', **styles)
        #self.efGraphIdx = self.frameInfo.addWidget(self.efGraph)

        self.bullEyeIdx = self.frameInfo.addWidget(self.bullEye)
        self.frameInfo.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)

        self.dataLayout.addWidget(self.frameInfo, 1, 1, 1, 1, alignment=Qt.AlignCenter)

        self.infoLayout = QVBoxLayout()
        self.infoLayout.setObjectName(u"infoLayout")

        self.folderView = QTreeView()
        self.folderView.setObjectName(u"folderView")
        self.folderView.setMinimumSize(QSize(200, 0))
        self.folderView.setMaximumSize(QSize(250, 1e4))
        self.infoLayout.addWidget(self.folderView)

        self.processButton = QPushButton()
        self.processButton.setObjectName(u"processButton")
        self.infoLayout.addWidget(self.processButton)

        self.changeFolderButton = QPushButton()
        self.changeFolderButton.setText("Change Folder")
        self.infoLayout.insertWidget(0, self.changeFolderButton)

        self.horizontalLayout.addLayout(self.infoLayout)
        self.horizontalLayout.addWidget(self.dataWidget)

        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Echocardiography", None))
        MainWindow.setGeometry(0, 0, 0, 0)

        self.processButton.setText(QCoreApplication.translate("MainWindow", u"Process", None))
