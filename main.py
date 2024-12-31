import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow,QAction, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QSlider, QDialog, QGridLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QPoint ,QRect
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

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.rois = []
    doubleClicked = pyqtSignal(QPoint)  # Signal to emit when label is double-clicked

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit(event.pos())
        super().mouseDoubleClickEvent(event)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and len(self.rois) < 2:  # Allow only two rectangles
            self.start_point = event.pos()
            self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end_point = event.pos()
            self.update()  # Trigger repaint to show the rectangle

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.end_point = event.pos()
            self.is_drawing = False

            # Calculate ROI as (x, y, w, h)
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.rois.append(roi)  # Add ROI to the list
            self.update()  # Final rectangle rendering

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)

            # Draw rectangles for all saved ROIs
            for roi in self.rois:
                x, y, w, h = roi
                painter.drawRect(QRect(x, y, w, h))

            # Draw the currently drawn rectangle
            if self.is_drawing:
                rect = QRect(self.start_point, self.end_point)
                painter.drawRect(rect)

    def get_rois(self):
        """Return the stored ROIs."""
        return self.rois

    def reset(self):
        """Clear all stored ROIs."""
        self.rois = []
        self.update()


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Processing")

        # Initialize images
        self.input_image = None
        self.original_image = None
        self.output1_image = None
        self.output2_image = None

        # Store the original image for zooming and processing
        self.processed_images = {
            "Input Viewport": None,
            "Output Viewport 1": None,
            "Output Viewport 2": None,
        }

        self.create_menu_bar()

        # Main Layout
        self.main_layout = QVBoxLayout()

        # Image Viewports Layout
        self.image_layout = QHBoxLayout()
        self.create_viewports()
        self.image_layout.addStretch()  # Centering the viewports
        self.main_layout.addLayout(self.image_layout)

        # Controls Layout
        self.controls_layout = QVBoxLayout()
        self.create_controls()

        self.calculate_snr_button = QPushButton("Calculate SNR")
        self.calculate_cnr_button = QPushButton("Calculate CNR")
        self.reset_button = QPushButton("Reset Rectangles")
        self.cnr_label = QLabel("CNR Value : NA")
        self.snr_label = QLabel("SNR Value : NA")
        self.controls_layout.addWidget(self.calculate_snr_button)
        self.controls_layout.addWidget(self.calculate_cnr_button)
        self.controls_layout.addWidget(self.reset_button)
        self.controls_layout.addWidget(self.snr_label)
        self.controls_layout.addWidget(self.cnr_label)
        self.calculate_snr_button.clicked.connect(self.calculate_snr)
        self.calculate_cnr_button.clicked.connect(self.calculate_cnr)
        self.reset_button.clicked.connect(self.reset_rectangles)
        self.main_layout.addLayout(self.controls_layout)

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        """Create the menu bar with actions."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        # Denoising Menu
        denoising_menu = menubar.addMenu("Denoising")
        self.create_denoising_actions(denoising_menu)

        # Noise Menu
        noise_menu = menubar.addMenu("Noise")
        self.create_noise_actions(noise_menu)

        # Filter Menu
        filter_menu = menubar.addMenu("Filters")
        self.create_filter_actions(filter_menu)

        # Histogram Menu
        histogram_menu = menubar.addMenu("Histogram")
        histogram_eq_action = QAction("Histogram Equalization", self)
        histogram_eq_action.triggered.connect(self.apply_histogram_equalization)
        histogram_menu.addAction(histogram_eq_action)

        clahe_action = QAction("CLAHE", self)
        clahe_action.triggered.connect(self.apply_clahe)
        histogram_menu.addAction(clahe_action)

        custom_contrast_action = QAction("Custom Contrast", self)
        custom_contrast_action.triggered.connect(self.apply_custom_contrast)
        histogram_menu.addAction(custom_contrast_action)

    def create_denoising_actions(self, menu):
        """Create denoising actions and add them to the menu."""
        denoise_action = QAction("Apply Denoising", self)
        denoise_action.triggered.connect(self.apply_denoising)
        menu.addAction(denoise_action)

    def create_noise_actions(self, menu):
        """Create noise actions and add them to the menu."""
        gaussian_noise_action = QAction("Add Gaussian Noise", self)
        gaussian_noise_action.triggered.connect(self.add_gaussian_noise)
        menu.addAction(gaussian_noise_action)

        salt_pepper_noise_action = QAction("Add Salt & Pepper Noise", self)
        salt_pepper_noise_action.triggered.connect(self.add_salt_pepper_noise)
        menu.addAction(salt_pepper_noise_action)

        poisson_noise_action = QAction("Add Poisson Noise", self)
        poisson_noise_action.triggered.connect(self.add_poisson_noise)
        menu.addAction(poisson_noise_action)

    def create_filter_actions(self, menu):
        """Create filter actions and add them to the menu."""
        low_pass_action = QAction("Apply LowPass Filter", self)
        low_pass_action.triggered.connect(self.apply_low_pass_filter)
        menu.addAction(low_pass_action)

        high_pass_action = QAction("Apply HighPass Filter", self)
        high_pass_action.triggered.connect(self.apply_high_pass_filter)
        menu.addAction(high_pass_action)

    def create_viewports(self):
        """Create and setup image viewports."""
        self.input_label = ClickableLabel("Input Viewport")
        self.output_label1 = ClickableLabel("Output Viewport 1")
        self.output_label2 = ClickableLabel("Output Viewport 2")

        for label in [self.input_label, self.output_label1, self.output_label2]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid rgb(9, 132, 227);; background-color: black;")
            label.setFixedSize(300, 300)

        self.image_layout.addWidget(self.input_label)
        self.image_layout.addWidget(self.output_label1)
        self.image_layout.addWidget(self.output_label2)

    def create_controls(self):
        """Create and setup controls for the application."""
        # Dropdown boxes for viewport selection
        self.edit_viewport_selector = self.create_combobox(
            ["Input Viewport", "Output Viewport 1", "Output Viewport 2"], "Edit Viewport:"
        )
        self.apply_viewport_selector = self.create_combobox(
            ["Input Viewport", "Output Viewport 1", "Output Viewport 2"], "Apply Changes To:"
        )
        self.zoom_technique = self.create_combobox(
            ["Nearest-Neighbor", "Linear", "Bi-linear","Cubic"], "Zoom Technique:"
        )
        # Sliders for zoom, brightness, and contrast
        self.zoom_slider = self.create_slider(Qt.Horizontal, 1, 4, 1, "Zoom (1x to 4x)")
        self.brightness_slider = self.create_slider(Qt.Horizontal, -100, 100, 0, "Brightness (-100 to 100)")
        self.contrast_slider = self.create_slider(Qt.Horizontal, 0, 200, 100, "Contrast (0 to 200)")

        # Additional options
        self.denoising_dropdown = self.create_combobox(
            ["Median Filter", "Gaussian Filter", "Non-Local Means"], "Denoising Method:"
        )

        # Connect signals to the respective slots
        self.input_label.doubleClicked.connect(lambda pos: self.show_histogram(pos, self.input_label))
        self.output_label1.doubleClicked.connect(lambda pos: self.show_histogram(pos, self.output_label1))
        self.output_label2.doubleClicked.connect(lambda pos: self.show_histogram(pos, self.output_label2))
        self.zoom_slider.valueChanged.connect(self.zoom_image)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness_contrast)
        self.contrast_slider.valueChanged.connect(self.adjust_brightness_contrast)

    def create_combobox(self, items, label_text):
        """Create a labeled combobox."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        combobox = QComboBox()
        combobox.addItems(items)
        layout.addWidget(combobox)
        self.controls_layout.addLayout(layout)
        return combobox

    def create_slider(self, orientation, min_value, max_value, initial_value, label_text):
        """Create a labeled slider."""
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label_text))
        slider = QSlider(orientation)
        slider.setRange(min_value, max_value)
        slider.setValue(initial_value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        layout.addWidget(slider)
        self.controls_layout.addLayout(layout)
        return slider

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
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)

        # Convert to grayscale for histogram equalization
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)

        # Convert back to 3-channel image for display
        result_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

        # Display result
        self.display_image(result_image, apply_label)

    def apply_clahe(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)

        # Convert to grayscale for CLAHE
        gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
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
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)

        # Apply Gamma Correction
        gamma = 1.5  # Change this value for different results
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        result_image = cv2.LUT(current_image, look_up_table)

        # Display result
        self.display_image(result_image, apply_label)
    def adjust_brightness_contrast(self):
        if self.original_image is None:
            print("No image loaded.")
            return
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)

        # Get brightness and contrast values
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()

        # Calculate the alpha (contrast) and beta (brightness)
        alpha = 1 + (contrast / 100.0)  # Scale factor
        beta = brightness  # Brightness addition

        # Apply the adjustments to the image
        adjusted_image = cv2.convertScaleAbs(current_image, alpha=alpha, beta=beta)


        self.display_image(adjusted_image, apply_label)

    def get_current_image(self, label):
        # Retrieve the current image from the QLabel
        pixmap = label.pixmap()
        if pixmap is not None:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Convert QImage to a NumPy array
            ptr = image.constBits()
            ptr.setsize(image.byteCount())
            img_array = np.array(ptr).reshape((height, width, 4))  # Assuming RGBA format

            # Convert to BGR format for OpenCV compatibility
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            return img_bgr
        return None
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
            selected_zoom_technique = self.zoom_technique.currentText()
            #["Nearest-Neighbor", "Linear", "Bi-linear", "Cubic"]
            if selected_zoom_technique == "Nearest-Neighbor":
                resized_image = cv2.resize(cropped_image, (label_width, label_height), interpolation=cv2.INTER_NEAREST)
            elif selected_zoom_technique == "Linear":
                resized_image = cv2.resize(cropped_image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
            elif selected_zoom_technique == "Bi-linear":
                resized_image = cv2.resize(cropped_image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
            elif  selected_zoom_technique == "Cubic":
                resized_image = cv2.resize(cropped_image, (label_width, label_height), interpolation=cv2.INTER_CUBIC)
        # Convert the resized image to QImage
        height, width, channel = resized_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)


        # Display the image in the QLabel
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

        label.setAlignment(Qt.AlignCenter)

    def zoom_image(self):


        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)
        if current_image is None:
            print("No image loaded.")
            return

        # Get the zoom factor from the slider
        factor = self.zoom_slider.value()
        print(f"Zoom Factor: {factor}")

        # Determine which viewport to edit


        # Use the image from the edit viewport and apply changes to the apply viewport
        edited_image = current_image  # Replace with logic to get the actual edited image from the edit viewport
        self.display_image(edited_image, apply_label, zoom_factor=factor)

    def add_gaussian_noise(self):
        if self.original_image is None:
            print("No image loaded.")
            return
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]



        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)

        mean = 0  # mean for Gaussian noise
        sigma = 25  # standard deviation for Gaussian noise

        # Check if current_image is loaded correctly and has the right dtype
        if current_image is None:
            raise ValueError("Current image is not loaded properly.")
        if current_image.dtype != np.uint8:
            raise ValueError("Current image should be of type uint8.")

        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, sigma, current_image.shape).astype(np.float32)

        # Apply noise, ensuring both images are of type float32
        noisy_image = cv2.add(current_image.astype(np.float32), gaussian_noise)

        # Clip the values to the uint8 range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)

        # Convert back to uint8
        noisy_image = noisy_image.astype(np.uint8)

        # Display result in the apply viewport
        self.display_image(noisy_image, apply_label)

    def add_salt_pepper_noise(self):
        if self.original_image is None:
            print("No image loaded.")
            return
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)
        # Add Salt-and-Pepper noise
        noisy_image = current_image
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
        self.display_image(noisy_image, apply_label)

    def apply_denoising(self):

        if self.original_image is None:
            return

        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)

        method = self.denoising_dropdown.currentText()

        noisy = current_image
        print(method)
        if method == "Median Filter":
            denoised = cv2.medianBlur(noisy, 5)
        elif method == "Gaussian Filter":
            denoised = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1)
        elif method == "Non-Local Means":
            denoised = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21)
        print("passed denoising")
        self.denoised_image = denoised
        self.display_image(denoised,apply_label)

    def add_poisson_noise(self):
        if self.original_image is None:
            print("No image loaded.")
            return

        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Copy the current image from the edit viewport
        current_image = self.get_current_image(edit_label)
        # Add Poisson noise
        noisy_image = np.random.poisson(current_image / 255.0 * 255).astype(np.uint8)

        # Display result
        self.display_image(noisy_image, apply_label)

    def apply_low_pass_filter(self):
        # Map selection to QLabel
        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }
        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Get the current image from the edit viewport
        current_image = self.get_current_image(edit_label)
        if current_image is None:
            print("No image loaded in the selected edit viewport.")
            return

        # Apply Gaussian Blur (Low-pass Filter)
        low_pass_image = cv2.GaussianBlur(current_image, (5, 5), 0)

        # Display result in the apply viewport
        self.display_image(low_pass_image, apply_label)

    # Apply High-pass Filter
    def apply_high_pass_filter(self):
        # Map selection to QLabel
        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        selected_edit_viewport = self.edit_viewport_selector.currentText()
        selected_apply_viewport = self.apply_viewport_selector.currentText()

        edit_label = viewport_mapping[selected_edit_viewport]
        apply_label = viewport_mapping[selected_apply_viewport]

        # Get the current image from the edit viewport
        current_image = self.get_current_image(edit_label)
        if current_image is None:
            print("No image loaded in the selected edit viewport.")
            return

        # Apply a high-pass filter using a kernel
        low_pass_image = cv2.GaussianBlur(current_image, (21, 21), 0)
        high_pass_image = cv2.subtract(current_image, low_pass_image)

        # Display result in the apply viewport
        self.display_image(high_pass_image, apply_label)

    def show_histogram(self, pos, label):

        selected_edit_viewport = self.edit_viewport_selector.currentText()

        viewport_mapping = {
            "Input Viewport": self.input_label,
            "Output Viewport 1": self.output_label1,
            "Output Viewport 2": self.output_label2,
        }

        edit_label = viewport_mapping[selected_edit_viewport]
        current_image = self.get_current_image(edit_label)

        if current_image is None:
            print("No image loaded.")
            return
        # Map QLabel coordinates to image coordinates
        label_width, label_height = label.width(), label.height()
        height, width, _ = current_image.shape

        x = int(pos.x() * width / label_width)
        y = int(pos.y() * height / label_height)

        # Ensure coordinates are within bounds
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)

        # Calculate histogram
        hist_r = cv2.calcHist([current_image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([current_image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([current_image], [2], None, [256], [0, 256])

        # Display histogram in a separate dialog
        histogram_dialog = HistogramDialog(hist_r, hist_g, hist_b, x, y, self)
        histogram_dialog.exec_()

    def calculate_snr(self):
        """Calculate SNR using the two selected ROIs."""
        if self.original_image is None:
            print("No image loaded.")
            return

        rois = self.input_label.get_rois()
        if len(rois) < 2:
            print("Please select two rectangles (signal and noise regions).")
            return

        # Extract the signal and noise regions
        signal_roi = rois[0]
        noise_roi = rois[1]

        x_s, y_s, w_s, h_s = signal_roi
        x_n, y_n, w_n, h_n = noise_roi

        signal_region = self.original_image[y_s:y_s + h_s, x_s:x_s + w_s]
        noise_region = self.original_image[y_n:y_n + h_n, x_n:x_n + w_n]

        # Convert to grayscale for processing
        signal_gray = cv2.cvtColor(signal_region, cv2.COLOR_BGR2GRAY)
        noise_gray = cv2.cvtColor(noise_region, cv2.COLOR_BGR2GRAY)

        # Calculate mean and std deviation
        mean_signal = np.mean(signal_gray)
        std_noise = np.std(noise_gray)

        if std_noise == 0:
            print("Noise standard deviation is zero. Cannot calculate SNR.")
            return

        snr = mean_signal / std_noise
        print(f"Calculated SNR: {snr:.2f}")
        new_snr_value = f"SNR Value : {snr:.2f}"
        self.snr_label.setText(new_snr_value)


    def calculate_cnr(self):
        """Calculate CNR using the two selected ROIs."""
        if self.original_image is None:
            print("No image loaded.")
            return

        rois = self.input_label.get_rois()
        if len(rois) < 2:
            print("Please select two rectangles (signal and noise regions).")
            return

        # Extract the signal and noise regions
        signal_roi = rois[0]
        noise_roi = rois[1]

        x_s, y_s, w_s, h_s = signal_roi
        x_n, y_n, w_n, h_n = noise_roi

        signal_region = self.original_image[y_s:y_s + h_s, x_s:x_s + w_s]
        noise_region = self.original_image[y_n:y_n + h_n, x_n:x_n + w_n]

        # Convert to grayscale for processing
        signal_gray = cv2.cvtColor(signal_region, cv2.COLOR_BGR2GRAY)
        noise_gray = cv2.cvtColor(noise_region, cv2.COLOR_BGR2GRAY)

        # Calculate mean and std deviation
        mean_signal = np.mean(signal_gray)
        mean_noise = np.mean(noise_gray)
        std_noise = np.std(noise_gray)

        if std_noise == 0:
            print("Noise standard deviation is zero. Cannot calculate CNR.")
            return

        cnr = abs(mean_signal - mean_noise) / std_noise

        new_cnr_value = f"CNR Value : {cnr:.2f}"
        print(f"Calculated CNR: {cnr:.2f}")
        self.cnr_label.setText(new_cnr_value)

    def reset_rectangles(self):
        """Reset rectangles on the image."""
        self.input_label.reset()


stylesheet = """ 
QWidget{ background-color: rgb(30,30,30);color: White;}
QLabel{ color: White;}
QPushButton {color: White; }
QTabWidget  {color: White; }

    QTextEdit {

               /* Blue text (rgb(9, 132, 227)) */
        border: 1px solid rgb(9, 132, 227);    /* Gray border */
    }


    QTextEdit QScrollBar::up-arrow:vertical, 
    QTextEdit QScrollBar::down-arrow:vertical {
        background: rgb(9, 132, 227);   /* Arrow color */

    }


"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(stylesheet)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
