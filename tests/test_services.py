import os
import unittest
import numpy as np
import cv2

from services.field_extractor import FieldExtractor
from services.document_detector import DocumentDetector
from services.image_preprocessor import ImagePreprocessor
from services.face_extractor import FaceExtractor
from services.ocr_service import OCRService


class TestServices(unittest.TestCase):
    def test_field_extractor_with_complete_id(self):
        sample_ocr = (
            "STATE OF CALIFORNIA\n"
            "DRIVER LICENSE\n"
            "NAME: JOHN DOE\n"
            "DOB: 15/08/1988\n"
            "EXP: 15/08/2029\n"
            "DL NO: D5678901\n"
            "CLASS: C\n"
        )
        extractor = FieldExtractor()
        fields = extractor.extract_fields(sample_ocr)

        self.assertEqual(fields["full_name"], "John Doe")
        self.assertEqual(fields["id_number"], "D5678901")
        self.assertEqual(fields["date_of_birth"], "15/08/1988")
        self.assertEqual(fields["expiry_date"], "15/08/2029")

    def test_field_extractor_with_heuristic_lines(self):
        sample_ocr = (
            "EMPLOYEE IDENTIFICATION\n"
            "ALICE JOHNSON\n"
            "ID # EMP-99881\n"
            "BIRTH DATE 1995-12-04\n"
            "VALID UNTIL 2027-12-31\n"
        )
        extractor = FieldExtractor()
        fields = extractor.extract_fields(sample_ocr)

        self.assertEqual(fields["full_name"], "Alice Johnson")
        self.assertEqual(fields["id_number"], "EMP-99881")
        self.assertEqual(fields["date_of_birth"], "1995-12-04")
        self.assertEqual(fields["expiry_date"], "2027-12-31")

    def test_field_extractor_missing_values_return_none(self):
        unreadable_text = "RANDOM TEXT WITHOUT ANY STANDARD ID INFORMATION OR NUMBERS"
        extractor = FieldExtractor()
        fields = extractor.extract_fields(unreadable_text)

        self.assertIsNone(fields["id_number"])
        self.assertIsNone(fields["date_of_birth"])
        self.assertIsNone(fields["expiry_date"])
        # Name should not hallucinate
        self.assertIsNone(fields["full_name"])

    def test_document_detector_detects_card_contour(self):
        # Create a background with a bright white rectangular card in the center
        bg = np.zeros((600, 800, 3), dtype=np.uint8)
        # Draw white ID card rectangle: top-left (150, 100), bottom-right (650, 500)
        cv2.rectangle(bg, (150, 100), (650, 500), (240, 240, 240), -1)

        detector = DocumentDetector(min_area_ratio=0.10)
        aligned, detected = detector.detect_and_align(bg)

        self.assertTrue(detected)
        self.assertIsNotNone(aligned)
        # Aligned card width should be close to 500, height close to 400
        h, w = aligned.shape[:2]
        self.assertTrue(450 <= w <= 550)
        self.assertTrue(350 <= h <= 450)

    def test_document_detector_fallback_on_flat_image(self):
        # Flat single-color image has no boundary
        flat = np.ones((400, 600, 3), dtype=np.uint8) * 128
        detector = DocumentDetector()
        result, detected = detector.detect_and_align(flat)

        self.assertFalse(detected)
        self.assertEqual(result.shape, flat.shape)

    def test_image_preprocessor_pipeline(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(img, "TEST", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess_for_ocr(img)

        self.assertIn("grayscale", processed)
        self.assertIn("denoised", processed)
        self.assertIn("adaptive_threshold", processed)
        self.assertIn("otsu_threshold", processed)
        self.assertEqual(len(processed["grayscale"].shape), 2)

    def test_face_extractor_no_face(self):
        # Image with plain geometry and text has no face
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 0), 2)

        face_extractor = FaceExtractor()
        path = face_extractor.extract_and_save(img, "test_no_face")
        self.assertIsNone(path)

    def test_face_extractor_with_face(self):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "sample_face.jpg")
        self.assertTrue(os.path.exists(asset_path), "Sample face asset should exist")
        face_img = cv2.imread(asset_path)
        self.assertIsNotNone(face_img)

        face_extractor = FaceExtractor()
        saved_path = face_extractor.extract_and_save(face_img, "test_detected_face")
        self.assertIsNotNone(saved_path)
        # Check that file actually exists on filesystem
        full_saved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), saved_path)
        self.assertTrue(os.path.exists(full_saved_path))
        # Check that saved thumbnail can be read
        thumbnail = cv2.imread(full_saved_path)
        self.assertIsNotNone(thumbnail)
        self.assertTrue(thumbnail.shape[0] > 0 and thumbnail.shape[1] > 0)

    def test_ocr_service_with_text(self):
        # Create a sharp text image
        img = np.ones((120, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "TEST OCR 123", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess_for_ocr(img)

        ocr_service = OCRService()
        extracted_text = ocr_service.extract_text(processed)
        self.assertIn("TEST", extracted_text)
        self.assertIn("123", extracted_text)


if __name__ == "__main__":
    unittest.main()

