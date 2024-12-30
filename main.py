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
        self.input_label = QLabel("Input Viewport")
        self.output_label1 = QLabel("Output Viewport 1")
        self.output_label2 = QLabel("Output Viewport 2")

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
        self.SNR_calc = QPushButton("SNR")

        self.controls_layout.addWidget(self.hist_eq_button)
        self.controls_layout.addWidget(self.clahe_button)
        self.controls_layout.addWidget(self.custom_contrast_button)
        self.controls_layout.addWidget(self.SNR_calc)
        self.main_layout.addLayout(self.controls_layout)
        self.hist_eq_button.clicked.connect(self.apply_histogram_equalization)
        self.SNR_calc.clicked.connect(self.select_rois_and_calculate_snr)
        self.clahe_button.clicked.connect(self.apply_clahe)
        self.custom_contrast_button.clicked.connect(self.apply_custom_contrast)
        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Connect buttons and sliders
        self.open_button.clicked.connect(self.open_image)



    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.processed_images["Input Viewport"] = self.original_image.copy()
            self.grayscaled_image = cv2.imread(self.original_image, cv2.IMREAD_GRAYSCALE)

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

    
    # Calculation of SNR of the image and ROIs

    def calculate_snr_from_rois(image, roi_signal, roi_noise):
        """
        Calculate the Signal-to-Noise Ratio (SNR) using two ROIs: signal and noise.

        Parameters:
            image (numpy.ndarray): Input image (grayscale).
            roi_signal (tuple): A tuple (x, y, w, h) representing the signal ROI.
            roi_noise (tuple): A tuple (x, y, w, h) representing the noise ROI.

        Returns:
            float: SNR value.
        """
        # Extract the signal ROI from the image
        x_s, y_s, w_s, h_s = roi_signal
        signal_roi_image = image[y_s:y_s + h_s, x_s:x_s + w_s]
        # Extract the noise ROI from the image
        x_n, y_n, w_n, h_n = roi_noise
        noise_roi_image = image[y_n:y_n + h_n, x_n:x_n + w_n]
        # Calculate the mean intensity (signal) in the signal ROI
        mean_signal = np.mean(signal_roi_image)
        # Calculate the standard deviation (noise) in the noise ROI
        noise = np.std(noise_roi_image)
        # Avoid division by zero
        if noise == 0:
            return float('inf')  # SNR is infinite if no noise
        # Compute SNR
        snr = mean_signal / noise
        return snr

    def select_rois_and_calculate_snr(image):
        """
        Allow the user to select two ROIs (signal and noise) and calculate the SNR.

        Parameters:
            image (numpy.ndarray): Input image (grayscale).
        """
        # Let the user select the signal ROI
        print("Select the signal ROI:")
        roi_signal = cv2.selectROI("Select Signal ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Signal ROI")  # Close the ROI selection window

        # Ensure the signal ROI is valid
        if roi_signal[2] == 0 or roi_signal[3] == 0:
            print("Invalid signal ROI selected.")
            return

        # Let the user select the noise ROI
        print("Select the noise ROI:")
        roi_noise = cv2.selectROI("Select Noise ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Noise ROI")  # Close the ROI selection window

        # Ensure the noise ROI is valid
        if roi_noise[2] == 0 or roi_noise[3] == 0:
            print("Invalid noise ROI selected.")
            return

        # Calculate SNR
        snr_value = calculate_snr_from_rois(image, roi_signal, roi_noise)
        

    






if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
