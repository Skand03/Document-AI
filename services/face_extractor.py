import os
import cv2
import numpy as np
from typing import Optional

DEFAULT_THUMBNAIL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "thumbnails"
)


class FaceExtractor:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or DEFAULT_THUMBNAIL_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_and_save(self, image: np.ndarray, record_id: str) -> Optional[str]:
        """
        Detects the most prominent portrait/face in the image,
        crops it with a portrait margin, saves to local storage, and returns the path.
        Returns None if no face is detected.
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(min(h, w) * 0.1), int(min(h, w) * 0.1)),
        )

        if len(faces) == 0:
            # Try with slightly looser parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30),
            )

        if len(faces) == 0:
            return None

        # Select the largest face by bounding area
        largest_face = max(faces, key=lambda r: r[2] * r[3])
        x, y, fw, fh = largest_face

        # Add portrait padding: 25% horizontally, 35% vertically (more on top for hair/head)
        pad_x = int(fw * 0.25)
        pad_y_top = int(fh * 0.35)
        pad_y_bottom = int(fh * 0.25)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y_bottom)

        face_roi = image[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        filename = f"{record_id}_face.jpg"
        save_path = os.path.join(self.output_dir, filename)
        
        # Save image as standard JPEG
        success = cv2.imwrite(save_path, face_roi)
        if success:
            # Return relative path for portability and clean API response
            rel_path = os.path.relpath(save_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return rel_path.replace("\\", "/")
        return None
