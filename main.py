import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QSlider, QDialog, QGridLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Processing")

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

        # Dropdown boxes
        self.edit_viewport_selector = QComboBox()
        self.edit_viewport_selector.addItems(["Input Viewport", "Output Viewport 1", "Output Viewport 2"])

        self.apply_viewport_selector = QComboBox()
        self.apply_viewport_selector.addItems(["Input Viewport", "Output Viewport 1", "Output Viewport 2"])

        self.controls_layout.addWidget(QLabel("Edit Viewport:"))
        self.controls_layout.addWidget(self.edit_viewport_selector)
        self.controls_layout.addWidget(QLabel("Apply Changes To:"))
        self.controls_layout.addWidget(self.apply_viewport_selector)

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
       # self.histogram_button.clicked.connect(self.show_histogram_in_ui)
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

    def display_image(self, image, label, zoom_factor=1):
        if image is None:
            label.clear()
            return

        # QLabel dimensions
        label_width, label_height = label.width(), label.height()

        if zoom_factor == 1:
            # For zoom factor 1, scale the entire image to fit QLabel
            resized_image = cv2.resize(image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
        else:
            # Calculate the region to crop based on the zoom factor
            height, width, _ = image.shape
            crop_width = width // zoom_factor
            crop_height = height // zoom_factor

            # Center the cropping area
            x1 = (width - crop_width) // 2
            y1 = (height - crop_height) // 2
            x2 = x1 + crop_width
            y2 = y1 + crop_height

            # Crop and resize to QLabel dimensions
            cropped_image = image[y1:y2, x1:x2]
            resized_image = cv2.resize(cropped_image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

        # Convert the resized image to QImage
        height, width, channel = resized_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display the image in the QLabel
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

    def zoom_image(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Get the zoom factor from the slider
        factor = self.zoom_slider.value()
        print(f"Zoom Factor: {factor}")

        # Determine which viewport to edit
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        # Map selection to QLabel
        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Use the image from the edit viewport and apply changes to the apply viewport
        edited_image = self.original_image  # Replace with logic to get the actual edited image from the edit viewport
        self.display_image(edited_image, apply_label, zoom_factor=factor)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
