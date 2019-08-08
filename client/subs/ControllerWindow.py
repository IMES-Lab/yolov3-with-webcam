from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from subs.ImageViewer import ImageViewer as ImageViewer


class ControllerWindow(QWidget):
    def __init__(self, target_object):
        super(ControllerWindow, self).__init__()

        self.target_fps = '15'
        self.target_resolution_scale = '1'

        self.index_performances = 0
        self.req_performances = []
        self.infer_performances = []
        self.fps_performances = []

        self.should_running = True

        self.targetYoloObject = target_object
        self.init_UI()

    def init_UI(self):
        self.setGeometry(200, 100, 1600, 900)
        self.sizeHint()
        self.setWindowTitle("Yolov3 with Webcam Controller")

        # Right Layout
        self.target_fps_lineEdit = QLineEdit('15')
        target_fps_pushButton = QPushButton("Set target fps")
        target_fps_pushButton.clicked.connect(lambda: self.set_target_fps())

        self.target_resolution_scale_lineEdit = QLineEdit('1')
        target_resolution_scale_pushButton = QPushButton("Set target resolution ratio")
        target_resolution_scale_pushButton.clicked.connect(lambda: self.set_target_resolution_scale())

        layout_target_fps = QVBoxLayout()
        layout_target_fps.addWidget(self.target_fps_lineEdit)
        layout_target_fps.addWidget(target_fps_pushButton)
        layout_target_fps.addSpacing(10)

        layout_target_resolution_scale = QVBoxLayout()
        layout_target_resolution_scale.addWidget(self.target_resolution_scale_lineEdit)
        layout_target_resolution_scale.addWidget(target_resolution_scale_pushButton)
        layout_target_fps.addSpacing(10)

        refresh_pushButton = QPushButton("Refresh performance data")
        refresh_pushButton.clicked.connect(lambda: self.refresh_data())

        self.startstop_pushButton = QPushButton("Stop")
        self.startstop_pushButton.clicked.connect(lambda: self.set_start_stop())

        self.image_viewer = ImageViewer()

        self.label_current_fps = QLabel('Currnet fps(estimated) : 0', self)
        self.label_current_fps.setText('Currnet fps(estimated) : 0')
        self.label_current_fps.setAlignment(Qt.AlignRight)

        layout_right = QVBoxLayout()
        layout_right.addWidget(self.image_viewer)
        layout_right.addWidget(self.label_current_fps)
        layout_right.addStretch(1)
        layout_right.addLayout(layout_target_fps)
        layout_right.addLayout(layout_target_resolution_scale)
        layout_right.addStretch(1)
        layout_right.addWidget(refresh_pushButton)
        layout_right.addStretch(3)
        layout_right.addWidget(self.startstop_pushButton)
        layout_right.addStretch(4)

        # Left Layout
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        layout_left = QVBoxLayout()
        layout_left.addWidget(self.canvas)
        self.canvas.setMinimumSize(630, 600)
        self.canvas.setMaximumSize(1000, 900)

        # Total Layout
        layout = QHBoxLayout()
        layout.addLayout(layout_left,2)
        layout.addLayout(layout_right,1)

        # Graph init

        self.ax_req = self.fig.add_subplot(311)
        self.ax_infer = self.fig.add_subplot(312)
        self.ax_fps = self.fig.add_subplot(313)

        self.ax_req.axis(ymin=0, ymax=0.6)
        self.ax_infer.axis(ymin=0, ymax=0.6)
        self.ax_fps.axis(ymin=0, ymax=12)

        self.req_line, = self.ax_req.plot([], [], label='req')
        self.infer_line, = self.ax_infer.plot([], [], label='infer')
        self.fps_line, = self.ax_fps.plot([], [], label='fps')

        self.ax_req.legend(loc='upper right')
        self.ax_infer.legend(loc='upper right')
        self.ax_fps.legend(loc='upper right')

        self.ax_req.grid()
        self.ax_infer.grid()
        self.ax_fps.grid()

        self.canvas.draw()

        self.setLayout(layout)

    def set_start_stop(self):
        if self.should_running:
            self.targetYoloObject.set_should_running(False)
            self.should_running = False
            self.startstop_pushButton.setText("Start")
        else:
            self.targetYoloObject.set_should_running(True)
            self.should_running = True
            self.startstop_pushButton.setText("Stop")

    def refresh_data(self):
        self.index_performances = 0
        self.req_performances = []
        self.infer_performances = []
        self.fps_performances = []

        self.ax_req.axis(ymin=0, ymax=0.6)
        self.ax_infer.axis(ymin=0, ymax=0.6)
        self.ax_fps.axis(ymin=0, ymax=12)

        self.canvas.draw()

    def set_target_fps(self):
        target_fps = int(self.target_fps_lineEdit.text())
        self.targetYoloObject.set_target_fps(target_fps)

    def set_target_resolution_scale(self):
        target_resolution_scale = float(self.target_resolution_scale_lineEdit.text())
        self.targetYoloObject.set_target_resolution_scale(target_resolution_scale)

    @pyqtSlot(int)
    def set_current_fps(self, current_fps):
        self.label_current_fps.setText('Currnet fps(estimated) : '+str(current_fps))

    @pyqtSlot(list)
    def add_performances(self, performance_list):
        # [request time, inference time, fps]
        self.index_performances += 1
        self.req_performances.append(performance_list[0])
        self.infer_performances.append(performance_list[1])
        self.fps_performances.append(performance_list[2])

        self.set_current_fps(performance_list[2])

        index_range = range(0, self.index_performances)

        # print(list(index_range))
        # print(self.req_performances)

        self.req_line.set_xdata(index_range)
        self.req_line.set_ydata(self.req_performances)
        self.infer_line.set_xdata(index_range)
        self.infer_line.set_ydata(self.infer_performances)
        self.fps_line.set_xdata(index_range)
        self.fps_line.set_ydata(self.fps_performances)

        self.ax_req.axis(xmin=0, xmax=self.index_performances+2, ymin=0, ymax=0.6)
        self.ax_infer.axis(xmin=0, xmax=self.index_performances+2,  ymin=0, ymax=0.6)
        self.ax_fps.axis(xmin=0, xmax=self.index_performances+2, ymin=0, ymax=max(self.fps_performances)+5)

        self.canvas.draw()