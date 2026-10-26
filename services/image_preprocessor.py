import cv2
import numpy as np
from typing import Dict, Any


class ImagePreprocessor:
    def __init__(self, target_dpi_scale: bool = True):
        self.target_dpi_scale = target_dpi_scale

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts BGR image to single-channel grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def reduce_noise(self, gray: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """Reduces image noise while preserving text edges."""
        if method == "bilateral":
            return cv2.bilateralFilter(gray, 9, 75, 75)
        elif method == "median":
            return cv2.medianBlur(gray, 3)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def apply_clahe(self, gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Applies adaptive Gaussian thresholding for binarization."""
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4
        )

    def otsu_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Applies Otsu's thresholding."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def preprocess_for_ocr(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Runs the preprocessing pipeline:
        1. Convert to grayscale.
        2. Upscale if image is too small (improves OCR on low-res text).
        3. CLAHE contrast enhancement (preserves character edges without binarization artifacts).
        4. Noise reduction.
        5. Adaptive thresholding and Otsu thresholding.
        Returns a dictionary of processed representations.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid")

        gray = self.to_grayscale(image)

        # Scale up if dimensions are too small for OCR readability
        h, w = gray.shape[:2]
        if self.target_dpi_scale and min(h, w) < 600:
            factor = 600.0 / min(h, w)
            gray = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)

        clahe_enhanced = self.apply_clahe(gray)
        denoised = self.reduce_noise(gray, method="gaussian")
        adaptive = self.adaptive_threshold(denoised)
        otsu = self.otsu_threshold(denoised)

        return {
            "grayscale": gray,
            "clahe": clahe_enhanced,
            "denoised": denoised,
            "adaptive_threshold": adaptive,
            "otsu_threshold": otsu,
        }
