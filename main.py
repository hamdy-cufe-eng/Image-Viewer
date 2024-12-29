import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QSlider, QDialog, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Viewer with Processing")

        # Initialize images
        self.input_image = None
        self.original_image = None  # Store the original image for zooming

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
            label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
            label.setFixedSize(300, 300)  # Fixed size for input and output labels

        self.image_layout.addWidget(self.input_label)
        self.image_layout.addWidget(self.output_label1)
        self.image_layout.addWidget(self.output_label2)
        self.main_layout.addLayout(self.image_layout)

        # Controls
        self.open_button = QPushButton("Open Image")
        self.histogram_button = QPushButton("Show Histogram")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 4)  # Zoom range from 1x to 4x
        self.zoom_slider.setValue(1)  # Default zoom level
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
        self.histogram_button.clicked.connect(self.show_histogram_in_ui)
        self.zoom_slider.valueChanged.connect(self.zoom_image)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Load image
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.input_image = self.original_image.copy()  # Store a copy for display
            self.display_image(self.input_image, self.input_label)  # Display the input image
            self.output_label1.clear()  # Clear output labels
            self.output_label2.clear()

    def display_image(self, image, label):
        if image is None:
            label.clear()
            return

        # Convert the image to QImage
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Create a QPixmap from QImage
        pixmap = QPixmap.fromImage(q_image)

        # Set the pixmap to the label with scaled size
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)  # Align the image in the center

    def show_histogram_in_ui(self):
        if self.input_image is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Histogram Viewer")
        layout = QGridLayout()

        gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_RGB2GRAY)
        plt.figure()
        plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title("Histogram")

        canvas = plt.gcf().canvas
        canvas.draw()

        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        histogram_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, channel = histogram_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(histogram_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        histogram_label = QLabel()
        histogram_label.setPixmap(pixmap)
        layout.addWidget(histogram_label)

        dialog.setLayout(layout)
        dialog.exec_()

    def zoom_image(self):
        if self.original_image is None:
            return

        factor = self.zoom_slider.value()
        new_height = int(self.original_image.shape[0] * factor)  # Calculate new height
        new_width = int(self.original_image.shape[1] * factor)  # Calculate new width

        # Resize the image to the new size
        zoomed_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        print(f"Zoom Factor: {factor}, New Width: {new_width}, New Height: {new_height}")

        # Display the zoomed image in Output Viewport 1
        self.display_image(zoomed_image, self.output_label1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
