from __future__ import division
import requests
import time
from utils.util import *
from utils.darknet import Darknet
import random
import pickle as pkl

import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from subs.custom_utils import *
from subs.ControllerWindow import ControllerWindow as ControllerWindow

url = 'http://210.102.181.242:11000/video_feed_data'

file_cfg = 'cfg/yolov3.cfg'
file_weights = 'weights/yolov3.weights'
num_classes = 80

confidence = 0.25
nms_thresh = 0.4
reso = 160

target_fps = 15
first_image = None


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

                if self.target_resolution_scale > 1:
                    frame = cv2.resize(frame, dsize=(0, 0), fx=self.target_resolution_scale,
                                       fy=self.target_resolution_scale, interpolation=cv2.INTER_CUBIC)
                elif 0 < self.target_resolution_scale < 1:
                    frame = cv2.resize(frame, dsize=(0, 0), fx=self.target_resolution_scale,
                                       fy=self.target_resolution_scale, interpolation=cv2.INTER_CUBIC)

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

            if self.should_running:
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
                # self.fps_signal.emit(fps)
                self.performance_signal.emit([req_time, infer_time, fps])
            else:
                pass

    def set_should_running(self, should_running):
        self.should_running = should_running

    def set_target_fps(self, para_target_fps):
        if para_target_fps is 0:
            self.target_fps = 999
        else:
            self.target_fps = para_target_fps

    def set_target_resolution_scale(self, target_resolution_scale):
        if target_resolution_scale < 0:
            self.target_resolution_scale = 0.1
        else:
            self.target_resolution_scale = target_resolution_scale


if __name__ == '__main__':
    app = QApplication(sys.argv)

    yolo_object = Yolov3WithCam()
    window = ControllerWindow(target_object=yolo_object)
    window.show()

    thread = QThread()
    yolo_object.moveToThread(thread)
    thread.started.connect(yolo_object.start_capture)
    thread.start()

    yolo_object.video_signal.connect(window.image_viewer.setImage)
    # yolo_object.fps_signal.connect(window.set_current_fps)
    yolo_object.performance_signal.connect(window.add_performances)

    app.exec_()
