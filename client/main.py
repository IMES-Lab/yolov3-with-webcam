from __future__ import division
import requests
import time
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from utils.util import *
from utils.darknet import Darknet
import random
import pickle as pkl

import threading

import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


url = 'http://210.102.181.242:11000/video_feed_data'

file_cfg = 'cfg/yolov3.cfg'
file_weights = 'weights/yolov3.weights'
num_classes = 80

confidence = 0.25
nms_thresh = 0.4
reso = 160

target_fps = 15
first_image = None


def get_frame_from_server(URL):
    res = requests.get(URL, stream=True)
    cnt = res.content
    decoded = cv2.imdecode(np.frombuffer(cnt, np.uint8), -1)

    return decoded


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setMinimumSize(600, 400)
        self.setMaximumSize(600, 400)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @pyqtSlot(QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")
        else:
            self.image = image.scaled(self.size())
            if image.size() != self.size():
                self.setMinimumSize(image.size())
            self.update()


class ControllerApp(QWidget):
    def __init__(self, target_object):
        super(ControllerApp, self).__init__()

        self.target_fps = '15'
        self.target_resolution_scale = '1'

        self.index_performances = 0
        self.req_performances = []
        self.infer_performances = []
        self.fps_performances = []

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

        target_fps_pushButton = QPushButton("Refresh performance data")
        target_fps_pushButton.clicked.connect(lambda: self.set_target_fps())

        self.image_viewer = ImageViewer()

        self.label_current_fps = QLabel('Currnet fps(estimated) : ??', self)
        self.label_current_fps.setText('Currnet fps(estimated) : ??')
        self.label_current_fps.setAlignment(Qt.AlignRight)

        layout_right = QVBoxLayout()
        layout_right.addWidget(self.image_viewer)
        layout_right.addWidget(self.label_current_fps)
        layout_right.addStretch(1)
        layout_right.addLayout(layout_target_fps)
        layout_right.addLayout(layout_target_resolution_scale)
        layout_right.addStretch(10)

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

        self.ax_req.axis(ymin=0, ymax=1.0)
        self.ax_infer.axis(ymin=0, ymax=1.0)
        self.ax_fps.axis(ymin=0, ymax=18)

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

    def refresh_data(self):
        self.index_performances = 0
        self.req_performances = []
        self.infer_performances = []
        self.fps_performances = []

        self.ax_req.axis(ymin=0, ymax=1.0)
        self.ax_infer.axis(ymin=0, ymax=1.0)
        self.ax_fps.axis(ymin=0, ymax=18)

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


class Yolov3WithCam(QObject):
    video_signal = pyqtSignal(QImage)
    fps_signal = pyqtSignal(int)
    performance_signal = pyqtSignal(list)

    def __init__(self):
        super(Yolov3WithCam, self).__init__()

        self.CUDA = None
        self.model = None
        self.inp_dim = None
        self.should_running = True

        self.target_fps = 15
        self.target_resolution_scale = 1

        self.first_frame = None

        self.init_model()

    def __del__(self):
        pass

    def init_model(self):
        self.CUDA = torch.cuda.is_available()
        self.model = Darknet(file_cfg)
        self.model.load_weights(file_weights)
        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])

        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        if self.CUDA:
            self.model.cuda()

        self.model.eval()

    def get_inferenced_frame(self, frame):
        img, orig_im, dim = prep_image(frame, self.inp_dim)

        if self.CUDA:
            img = img.cuda()

        output = self.model(Variable(img), self.CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh)

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim

        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("data/pallete", "rb"))

        list(map(lambda x: write(x, orig_im, classes, colors), output))

        return orig_im

    def start_capture(self):
        while True:
            try:
                req_start = time.time()

                frame = get_frame_from_server(URL=url)

                # print(self.target_resolution_scale)

                if self.target_resolution_scale > 1:
                    frame = cv2.resize(frame, dsize=(0, 0), fx=self.target_resolution_scale,
                                       fy=self.target_resolution_scale, interpolation=cv2.INTER_CUBIC)
                elif 0 < self.target_resolution_scale < 1:
                    frame = cv2.resize(frame, dsize=(0, 0), fx=self.target_resolution_scale,
                                       fy=self.target_resolution_scale, interpolation=cv2.INTER_CUBIC)
                # print(frame.shape)
                req_end = time.time()
                req_time = req_end - req_start

                infer_start = time.time()
            except Exception as e:
                print("Error while requesting")
                print(e)

            try:
                if self.should_running:

                    inferenced_frame = self.get_inferenced_frame(frame)

                    color_swapped_image = cv2.cvtColor(inferenced_frame, cv2.COLOR_BGR2RGB)
                    color_swapped_image = cv2.resize(color_swapped_image, dsize=(600, 400), interpolation=cv2.INTER_CUBIC)

                    qt_image = QImage(color_swapped_image.data,
                                             600,
                                             400,
                                             color_swapped_image.strides[0],
                                             QImage.Format_RGB888)

                    self.video_signal.emit(qt_image)
                else:
                    color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    color_swapped_image = cv2.resize(color_swapped_image, dsize=(600, 400),
                                                     interpolation=cv2.INTER_CUBIC)

                    qt_image = QImage(color_swapped_image.data,
                                      600,
                                      400,
                                      color_swapped_image.strides[0],
                                      QImage.Format_RGB888)
                    self.video_signal.emit(qt_image)
            except Exception as e:
                print("Error while inference")
                print(e)

            infer_end = time.time()
            infer_time = infer_end - infer_start

            target_delay = (1.0 / self.target_fps) - infer_time - req_time
            if target_delay < 0:
                target_delay = 0

            time.sleep(target_delay)

            job_end = time.time()
            job_time = job_end - req_start

            fps = round(1 / job_time)
            # print(job_time)
            self.fps_signal.emit(fps)
            self.performance_signal.emit([req_time, infer_time, fps])

    def set_target_fps(self, target_fps):
        if target_fps is 0:
            self.target_fps = 999
        else:
            self.target_fps = target_fps

    def set_target_resolution_scale(self, target_resolution_scale):
        if target_resolution_scale < 0:
            self.target_resolution_scale = 0.1
        else:
            self.target_resolution_scale = target_resolution_scale


if __name__ == '__main__':
    app = QApplication(sys.argv)

    yolo_object = Yolov3WithCam()
    window = ControllerApp(target_object=yolo_object)
    window.show()

    thread = QThread()
    yolo_object.moveToThread(thread)
    thread.started.connect(yolo_object.start_capture)
    thread.start()

    yolo_object.video_signal.connect(window.image_viewer.setImage)
    yolo_object.fps_signal.connect(window.set_current_fps)
    yolo_object.performance_signal.connect(window.add_performances)

    app.exec_()
