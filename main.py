import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QSlider)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Processing")

        # Initialize images
        self.input_image = None
        self.output_image1 = None
        self.output_image2 = None

        # Main layout
        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.controls_layout = QVBoxLayout()

        # Input and output viewports
        self.input_label = QLabel("Input Viewport")
        self.output_label1 = QLabel("Output Viewport 1")
        self.output_label2 = QLabel("Output Viewport 2")

        for label in [self.input_label, self.output_label1, self.output_label2]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black;")
            label.setFixedSize(300, 300)

        self.image_layout.addWidget(self.input_label)
        self.image_layout.addWidget(self.output_label1)
        self.image_layout.addWidget(self.output_label2)
        self.main_layout.addLayout(self.image_layout)

        # Controls
        self.open_button = QPushButton("Open Image")
        self.histogram_button = QPushButton("Show Histogram")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 4)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(1)

        self.controls_layout.addWidget(self.open_button)
        self.controls_layout.addWidget(self.histogram_button)
        self.controls_layout.addWidget(QLabel("Zoom (1x to 4x)"))
        self.controls_layout.addWidget(self.zoom_slider)

        self.main_layout.addLayout(self.controls_layout)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Connect buttons
        self.open_button.clicked.connect(self.open_image)
        self.histogram_button.clicked.connect(self.show_histogram)
        self.zoom_slider.valueChanged.connect(self.zoom_image)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.input_image, self.input_label)

    def display_image(self, image, label):
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def show_histogram(self):
        if self.input_image is None:
            return
        plt.figure("Histogram")
        plt.hist(self.input_image.ravel(), bins=256, range=(0, 256), color='gray')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title("Histogram")
        plt.show()

    def zoom_image(self):
        if self.input_image is None:
            return
        factor = self.zoom_slider.value()
        zoomed_image = cv2.resize(self.input_image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        self.display_image(zoomed_image, self.input_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
