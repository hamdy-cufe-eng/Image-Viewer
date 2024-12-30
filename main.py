import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QSlider, QDialog, QGridLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import matplotlib.pyplot as plt
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
class HistogramDialog(QDialog):
    def __init__(self, red_hist, green_hist, blue_hist, x, y, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Histogram at ({x}, {y})")
        self.resize(600, 400)


        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)

        self.plot_histogram(red_hist, green_hist, blue_hist)

    def plot_histogram(self, red_hist, green_hist, blue_hist):
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        ax.plot(red_hist, color="r", label="Red")
        ax.plot(green_hist, color="g", label="Green")
        ax.plot(blue_hist, color="b", label="Blue")
        ax.set_title("RGB Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        self.canvas.draw()

class ClickableLabel(QLabel):
    doubleClicked = pyqtSignal(QPoint)  # Signal to emit when label is double-clicked

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit(event.pos())
        super().mouseDoubleClickEvent(event)

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Processing")

        # Initialize images
        self.input_image = None
        self.original_image = None  # Store the original image for zooming and processing
        self.processed_images = {
            "Input Viewport": None,
            "Output Viewport 1": None,
            "Output Viewport 2": None,
        }

        # Main layout
        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.controls_layout = QVBoxLayout()

        # Input and output viewports
        self.input_label = ClickableLabel("Input Viewport")
        self.output_label1 = ClickableLabel("Output Viewport 1")
        self.output_label2 = ClickableLabel("Output Viewport 2")

        for label in [self.input_label, self.output_label1, self.output_label2]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
            label.setFixedSize(300, 300)

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
        self.zoom_slider.setRange(1, 4)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(1)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.zoom_image)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness_contrast)
        self.contrast_slider.valueChanged.connect(self.adjust_brightness_contrast)
        self.controls_layout.addWidget(self.open_button)
        self.controls_layout.addWidget(self.histogram_button)
        self.controls_layout.addWidget(QLabel("Zoom (1x to 4x)"))
        self.controls_layout.addWidget(self.zoom_slider)
        self.controls_layout.addWidget(QLabel("Brightness (-100 to 100)"))
        self.controls_layout.addWidget(self.brightness_slider)
        self.controls_layout.addWidget(QLabel("Contrast (0 to 200)"))
        self.controls_layout.addWidget(self.contrast_slider)

        self.hist_eq_button = QPushButton("Histogram Equalization")
        self.clahe_button = QPushButton("CLAHE")
        self.custom_contrast_button = QPushButton("Custom Contrast")
        self.gaussian_noise_button = QPushButton("Add Gaussian Noise")
        self.salt_pepper_noise_button = QPushButton("Add Salt-and-Pepper Noise")
        self.poisson_noise_button = QPushButton("Add Poisson Noise")

        self.controls_layout.addWidget(self.gaussian_noise_button)
        self.controls_layout.addWidget(self.salt_pepper_noise_button)
        self.controls_layout.addWidget(self.poisson_noise_button)
        self.controls_layout.addWidget(self.hist_eq_button)
        self.controls_layout.addWidget(self.clahe_button)
        self.controls_layout.addWidget(self.custom_contrast_button)
        self.main_layout.addLayout(self.controls_layout)
        self.hist_eq_button.clicked.connect(self.apply_histogram_equalization)
        self.clahe_button.clicked.connect(self.apply_clahe)
        self.custom_contrast_button.clicked.connect(self.apply_custom_contrast)
        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Connect buttons and sliders
        self.open_button.clicked.connect(self.open_image)
        self.input_label.doubleClicked.connect(lambda pos: self.show_histogram(pos, self.input_label))
        self.output_label1.doubleClicked.connect(lambda pos: self.show_histogram(pos, self.output_label1))
        self.output_label2.doubleClicked.connect(lambda pos: self.show_histogram(pos, self.output_label2))

        self.gaussian_noise_button.clicked.connect(self.add_gaussian_noise)
        self.salt_pepper_noise_button.clicked.connect(self.add_salt_pepper_noise)
        self.poisson_noise_button.clicked.connect(self.add_poisson_noise)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.processed_images["Input Viewport"] = self.original_image.copy()
            self.display_image(self.original_image, self.input_label)

    def apply_histogram_equalization(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Convert to grayscale for histogram equalization
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)

        # Convert back to 3-channel image for display
        result_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

        # Display result
        self.display_image(result_image, self.output_label1)

    def apply_clahe(self):
        if self.original_image is None:
            print("No image loaded.")
            return

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

        # Convert to grayscale for CLAHE
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray_image)

        # Convert back to 3-channel image for display
        result_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)

        # Display result
        self.display_image(result_image, apply_label)

        def apply_custom_contrast(self):
            if self.original_image is None:
                print("No image loaded.")
                return

            # Apply Gamma Correction
            gamma = 1.5  # Change this value for different results
            look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
            result_image = cv2.LUT(self.original_image, look_up_table)

            # Display result
            self.display_image(result_image, self.output_label2)

    def apply_custom_contrast(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Apply Gamma Correction
        gamma = 1.5  # Change this value for different results
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        result_image = cv2.LUT(self.original_image, look_up_table)

        # Display result
        self.display_image(result_image, self.output_label2)
    def adjust_brightness_contrast(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Get brightness and contrast values
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()

        # Calculate the alpha (contrast) and beta (brightness)
        alpha = 1 + (contrast / 100.0)  # Scale factor
        beta = brightness  # Brightness addition

        # Apply the adjustments to the image
        adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)

        # Determine which viewport to edit and apply changes
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()
#
        # Map selection to QLabel
        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        apply_label = viewport_mapping[selected_apply_viewport]
        self.display_image(adjusted_image, apply_label)

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

    def add_gaussian_noise(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Add Gaussian noise
        mean = 0
        stddev = 25
        gaussian_noise = np.random.normal(mean, stddev, self.original_image.shape).astype(np.uint8)
        noisy_image = cv2.add(self.original_image, gaussian_noise)

        # Display result
        self.display_image(noisy_image, self.output_label1)

    def add_salt_pepper_noise(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Add Salt-and-Pepper noise
        noisy_image = self.original_image.copy()
        prob = 0.02  # Probability of noise
        threshold = 1 - prob
        for i in range(noisy_image.shape[0]):
            for j in range(noisy_image.shape[1]):
                rand = np.random.rand()
                if rand < prob:
                    noisy_image[i, j] = 0  # Salt
                elif rand > threshold:
                    noisy_image[i, j] = 255  # Pepper

        # Display result
        self.display_image(noisy_image, self.output_label2)

    def add_poisson_noise(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Add Poisson noise
        noisy_image = np.random.poisson(self.original_image / 255.0 * 255).astype(np.uint8)

        # Display result
        self.display_image(noisy_image, self.output_label2)

    def show_histogram(self, pos, label):
        if self.original_image is None:
            print("No image loaded.")
            return

        # Map QLabel coordinates to image coordinates
        label_width, label_height = label.width(), label.height()
        height, width, _ = self.original_image.shape

        x = int(pos.x() * width / label_width)
        y = int(pos.y() * height / label_height)

        # Ensure coordinates are within bounds
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)

        # Calculate histogram
        hist_r = cv2.calcHist([self.original_image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([self.original_image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([self.original_image], [2], None, [256], [0, 256])

        # Display histogram in a separate dialog
        histogram_dialog = HistogramDialog(hist_r, hist_g, hist_b, x, y, self)
        histogram_dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
