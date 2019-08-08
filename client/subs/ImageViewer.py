from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


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
