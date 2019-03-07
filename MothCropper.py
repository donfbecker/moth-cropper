from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import numpy
import os
import shutil
import sys

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

def isPinned(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pinned = (numpy.average(gray) < 127)
    gray = None
    return pinned

def getPinnedBBox(img):
    contrast = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.Laplacian(contrast, cv2.CV_64F)
    contrast = (255 - contrast)
    contrast = cv2.threshold(contrast, 127, 255, cv2.THRESH_BINARY)[1]

    mask = contrast < 200
    coords = numpy.argwhere(mask)

    # Bounding box of non-black pixels.
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1
    return top, left, bottom, right

def getSheetBBox(img, radius):
    white = [255, 255, 255]
    mask  = [0, 0, 255]

    kernel = numpy.array([
        [16, 32, 16],
        [32, 64, 32],
        [16, 32, 16]
    ]) / 128
    contrast = cv2.filter2D(img, -1, kernel=kernel)

    #contrast = (img - 0.5) * 2 + 0.5
    #contrast = int(contrast)

    height, width = contrast.shape[:2]
    hx = int(width / 2)
    hy = int(height / 2)

    if width > height:
        search_radius = int(height / 3)
    else:
        search_radius = int(width / 3)

    # look for non-white pixels nearest to the center
    def findObjectNearCenter():
        for r in range(1, search_radius):
            for x in range(hx - r, hx + r):
                for y in (hy - r, hy + r):
                    px = contrast[y, x]
                    if not all(px == white):
                        return x,y

    x,y = findObjectNearCenter()

    top, left, bottom, right = (height, width, 0, 0)

    queue = [
        (x, y),
        (x, y - 10),
        (x, y + 10),
        (x - 10, y),
        (x + 10, y)
    ]

    r = radius
    while len(queue) > 0:
        x,y = queue.pop()
        if all(contrast[y, x] == mask) or all(contrast[y, x] == white):
            continue

        contrast[y, x] = mask

        if y < top:
            top = y
        if y > bottom:
            bottom = y
        if x < left:
            left = x
        if x > right:
            right = x

        for cx in range(max(x - r, 1), min(x + r, width - 1)):
           for cy in range(max(y - r, 1), min(y + r, height - 1)):
               cc = contrast[cy, cx]
               if not (all(cc == white) or all(cc == mask)):
                   queue.append((cx, cy))

    return top, left, bottom, right

def crop_image(filename, radius=3, padding=0.05):
    img = cv2.imread(filename)
    full_height, full_width = img.shape[:2]

    # Make the image smaller, less memory to work with
    if full_width > full_height:
        scale = 500 / float(full_width)
    else:
        scale = 500 / float(full_height)

    #scale = 0.25
    img = cv2.resize(img, None, fx=scale, fy=scale)
    height, width = img.shape[:2]

    if isPinned(img):
        top, left, bottom, right = getPinnedBBox(img)
    else:
        top, left, bottom, right = getSheetBBox(img, radius)

    # reload the original image at full size
    img = cv2.imread(filename)
    height, width = img.shape[:2]

    # scale the box up to the original size
    top = int(top / scale)
    left = int(left / scale)
    bottom = int(bottom / scale)
    right = int(right / scale)

    box_width = right - left
    box_height = bottom - top

    if box_width > box_height:
        pad = (box_width - box_height) / 2;
        top = max(0, top - pad)
        bottom = top + box_width
        box_height = box_width
    else:
        pad = (box_height - box_width) / 2;
        left = max(0, left - pad)
        right = left + box_height
        box_width = box_height

    pad = int(box_width * padding)
    if (box_width + (pad * 2)) > width:
        pad = (width - box_width) / 2
    if (box_height + (pad * 2)) > height:
        pad = (height - box_height) / 2

    top = top - pad;
    left = left - pad;
    bottom = bottom + pad;
    right = right + pad;

    box_width = right - left
    box_height = bottom - top

    if top < 0:
        top = 0
        bottom = box_height
    if left < 0:
        left = 0
        right = box_width
    if bottom > height:
        bottom = height
        top = bottom - box_height
    if right > width:
        right = width
        left = right - box_width

    top = int(top)
    left = int(left)
    bottom = int(bottom)
    right = int(right)

    crop = img[top:bottom, left:right]
    crop = cv2.resize(crop, (600, 600))

    crop_path = os.path.splitext(filename)[0] + "-cropped.jpg"
    cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return crop_path

class CropDirectoryThread(QThread):
    count = pyqtSignal(int)
    display = pyqtSignal('QString')
    progress = pyqtSignal(int)
    complete = pyqtSignal()

    def __init__(self, path, radius, padding):
        QThread.__init__(self)
        self.path = path
        self.radius = radius
        self.padding = padding
        self.running = True

    def __del__(self):
        self.wait()

    def stop(self):
        self.running = False

    def run(self):
        files = glob.glob(os.path.join(self.path, "*.bmp"))
        files.extend(glob.glob(os.path.join(self.path, "*.gif")))
        files.extend(glob.glob(os.path.join(self.path, "*.jpg")))
        files.extend(glob.glob(os.path.join(self.path, "*.jpeg")))
        files.extend(glob.glob(os.path.join(self.path, "*.png")))

        self.count.emit(len(files))

        i = 0
        for file in files:
            if not self.running:
                break

            if(os.path.isfile(file) and not file.endswith('-cropped.jpg')):
                result = crop_image(file, self.radius, self.padding)
                self.display.emit(result)

            self.progress.emit(i)
            i += 1

        self.complete.emit()
        print("CropDirectoryThread finished.")

class CropImageThread(QThread):
    display = pyqtSignal('QString')
    complete = pyqtSignal()

    def __init__(self, path, radius, padding):
        QThread.__init__(self)
        self.path = path
        self.radius = radius
        self.padding = padding

    def __del__(self):
        self.wait()

    def run(self):
        result = crop_image(self.path, self.radius, self.padding)
        self.display.emit(result)
        self.complete.emit()

class ImageLabel(QLabel):
    def __init__(self, image):
        super(ImageLabel, self).__init__("")
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setImage(image)

    def setImage(self, image):
        self.pixmap = QPixmap(image).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.pixmap)

    def width(self):
        return self.pixmap.width()

    def height(self):
        return self.pixmap.height()

class MothCropper(QMainWindow):
    progdialog = None
    thread = None

    def __init__(self):
        super(MothCropper, self).__init__()
        self.setAcceptDrops(True)
        self.initUI()

    def closeEvent(self, e):
        self.quit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        path = str(e.mimeData().urls()[0].toLocalFile())

        if os.path.isdir(path):
            self.cropDirectory(path)
        else:
            self.cropAndDisplayImage(path)

    def quit(self):
        app.quit()
        sys.exit()

    def initUI(self):
        self.setWindowTitle("Moth Cropper")
        self.setWindowIcon(QIcon('moth.ico'))
        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QGridLayout()
        widget.setLayout(layout)

        fileAct = QAction('Crop &Image', self)
        fileAct.setShortcut('Ctrl+I')
        fileAct.setStatusTip('Crop a single image')
        fileAct.triggered.connect(self.cropImage)

        directoryAct = QAction('Crop &Directory', self)
        directoryAct.setShortcut('Ctrl+D')
        directoryAct.setStatusTip('Crop all images in a directory')
        directoryAct.triggered.connect(self.menuCropDirectory)

        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(fileAct)
        fileMenu.addAction(directoryAct)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAct)

        self.label = ImageLabel("drop.png");
        layout.addWidget(self.label, 1, 1, 1, 7)

        l = QLabel()
        l.setText("Radius:")
        layout.addWidget(l, 2, 2, 1, 1)

        self.radius = QSpinBox()
        self.radius.setMinimum(1)
        self.radius.setMaximum(10)
        self.radius.setValue(3)
        layout.addWidget(self.radius, 2, 3)

        l = QLabel()
        l.setText("Padding %:")
        layout.addWidget(l, 2, 5, 1, 1)

        self.padding = QSpinBox()
        self.padding.setMinimum(0)
        self.padding.setMaximum(100)
        self.padding.setValue(5)
        layout.addWidget(self.padding, 2, 6)

        self.resizeUI()

        rect = self.frameGeometry()
        center = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())

        self.show()

    def resizeUI(self):
        self.resize(self.label.width(), self.label.height() + self.menuBar().height());

    def displayImage(self, path):
        self.label.setImage(path)
        self.resizeUI()

    def cropAndDisplayImage(self, path):
        self.displayImage(path)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        radius = self.radius.value()
        padding = self.padding.value() / 100

        self.thread = CropImageThread(path, radius, padding)
        self.thread.display.connect(self.displayImage)
        self.thread.complete.connect(self.signalComplete)
        self.thread.start()
        return

    def signalCancel(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        if self.progdialog:
            self.progdialog.close()
            self.progdialog = None

    def signalImageCount(self, count):
        if self.progdialog:
            self.progdialog.setMaximum(count)

    def signalProgress(self, value):
        if self.progdialog:
            self.progdialog.setValue(value)

    def signalComplete(self):
        if self.progdialog:
            self.progdialog.close()
        QApplication.restoreOverrideCursor()

    def cropImage(self):
        path = QFileDialog.getOpenFileName(self, "Choose an image", "", "Images (*.bmp *.gif *.jpg *.jpeg *.png)")
        if(path[0]):
            self.cropAndDisplayImage(path[0])
        return

    def cropDirectory(self, path):
        radius = self.radius.value()
        padding = self.padding.value() / 100

        self.progdialog = QProgressDialog("", "Cancel", 0, 100, self)
        self.progdialog.setWindowTitle("Cropping")
        self.progdialog.setWindowModality(Qt.WindowModal)
        self.progdialog.canceled.connect(self.signalCancel)
        self.progdialog.show()

        self.thread = CropDirectoryThread(path, radius, padding)
        self.thread.count.connect(self.signalImageCount)
        self.thread.display.connect(self.displayImage)
        self.thread.progress.connect(self.signalProgress)
        self.thread.complete.connect(self.signalComplete)
        self.thread.start()

    def menuCropDirectory(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a directory", "", QFileDialog.ShowDirsOnly)
        if(path):
            self.cropDirectory(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MothCropper()
    sys.exit(app.exec_())
