import io
import os
import tempfile
import unittest
import numpy as np
import cv2

from services.field_extractor import FieldExtractor
from services.document_detector import DocumentDetector
from services.image_preprocessor import ImagePreprocessor
from services.ocr_service import OCRService
from app import create_app


class TestOCRAndFieldExtractionImprovements(unittest.TestCase):
    def setUp(self):
        self.extractor = FieldExtractor()

    def test_institutional_id_card_case(self):
        """
        Tests the actual failing case:
        - University header with FACULTY OF ENGINEERING AND TECHNOLOGY MIT-WPU
        - Human name: Chaubey Skand
        - Course: MCA
        - Student ID / PRN: 1272251121
        - Emergency phone number: 917905978260 (must not be confused with ID)
        - Issue and Expiry: Issued on 07/2026 Valid Upto; 07/2027
        - Date of Birth: Absent (must return None, never hallucinate)
        """
        mit_card_text = (
            "FACULTY OF\n"
            "ENGINEERING AND\n"
            "TECHNOLOGY\n"
            "MIT-WPU\n"
            "Chaubey Skand -\n"
            "MCA\n"
            "UNL\n"
            "1272251121\n"
            "Blood Group: B+\n"
            "Emergency No: 917905978260\n"
            "Issued on 07/2026 Valid Upto; 07/2027\n"
        )
        fields = self.extractor.extract_fields(mit_card_text)

        self.assertEqual(fields["full_name"], "Chaubey Skand", "Full name must be correctly extracted without picking university header")
        self.assertEqual(fields["id_number"], "1272251121", "ID number must be extracted and not confused with Emergency No")
        self.assertEqual(fields["expiry_date"], "07/2027", "Expiry date must be extracted from Valid Upto")
        self.assertIsNone(fields["date_of_birth"], "DOB must be None when absent on card")

    def test_multiline_labeled_dates_and_alphanumeric_pan(self):
        """Tests multiline label layouts and alphanumeric PAN card format."""
        pan_text = (
            "INCOME TAX DEPARTMENT\n"
            "GOVT OF INDIA\n"
            "Permanent Account Number Card\n"
            "CRMPC6176N\n"
            "Name:\n"
            "Skand Chaubey\n"
            "Father's Name:\n"
            "Abhishek Chaubey\n"
            "Date of Birth:\n"
            "06/03/2004\n"
            "Signature\n"
        )
        fields = self.extractor.extract_fields(pan_text)

        self.assertEqual(fields["full_name"], "Skand Chaubey")
        self.assertEqual(fields["id_number"], "CRMPC6176N")
        self.assertEqual(fields["date_of_birth"], "06/03/2004")
        self.assertIsNone(fields["expiry_date"], "PAN cards do not expire; expiry must be None")

    def test_ambiguous_dates_without_labels_return_none(self):
        """
        When dates are present without explicit field labels,
        the system must NOT guess or classify the earliest/latest as DOB or expiry.
        """
        ambiguous_text = (
            "RANDOM BADGE\n"
            "John Doe\n"
            "Access Level 3\n"
            "14/05/2023\n"
            "18/09/2028\n"
        )
        fields = self.extractor.extract_fields(ambiguous_text)

        self.assertEqual(fields["full_name"], "John Doe")
        self.assertIsNone(fields["date_of_birth"], "Unlabeled date must not be assumed as DOB")
        self.assertIsNone(fields["expiry_date"], "Unlabeled date must not be assumed as Expiry")

    def test_phone_and_emergency_numbers_not_confused_with_id(self):
        """Emergency and contact numbers must not be treated as IDs."""
        text_with_phone = (
            "VISITOR PASS\n"
            "Holder: Alice Walker\n"
            "Emergency Contact: 9876543210\n"
            "Tel: 0112345678\n"
        )
        fields = self.extractor.extract_fields(text_with_phone)

        self.assertEqual(fields["full_name"], "Alice Walker")
        self.assertIsNone(fields["id_number"], "Phone numbers must not be classified as ID numbers")

    def test_inner_square_contour_rejected_by_detector(self):
        """
        A square feature inside an image (such as an ID photo box or QR code, ratio ~1.0)
        must NOT be mistaken for the document boundary.
        """
        detector = DocumentDetector(min_area_ratio=0.10)
        # Create a 1000x800 background with a square box (300x300, ratio 1.0)
        bg = np.zeros((800, 1000, 3), dtype=np.uint8)
        cv2.rectangle(bg, (350, 250), (650, 550), (220, 220, 220), -1)

        warped, detected = detector.detect_and_align(bg)
        self.assertFalse(detected, "Square inner box must not be selected as ID card boundary")
        self.assertEqual(warped.shape, bg.shape)

    def test_valid_rectangular_card_detected_by_detector(self):
        """A rectangular card (aspect ratio ~1.6) is detected and warped."""
        detector = DocumentDetector(min_area_ratio=0.10)
        bg = np.zeros((900, 1200, 3), dtype=np.uint8)
        # Rectangular card: width=640, height=400 (aspect ratio 1.6)
        cv2.rectangle(bg, (280, 250), (920, 650), (240, 240, 240), -1)

        warped, detected = detector.detect_and_align(bg)
        self.assertTrue(detected, "Standard rectangular ID card should be detected")
        self.assertTrue(380 <= warped.shape[0] <= 420)
        self.assertTrue(620 <= warped.shape[1] <= 660)

    def test_api_full_flow_with_institutional_fields(self):
        """Verifies full API execution with institutional card data via test client."""
        temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp_dir.name, "test_api_inst.db")
        thumb_dir = os.path.join(temp_dir.name, "thumbs")

        app = create_app({
            "TESTING": True,
            "DATABASE_PATH": db_path,
            "THUMBNAIL_DIR": thumb_dir,
        })
        client = app.test_client()

        # Create a synthetic vertical badge
        card_img = np.ones((700, 450, 3), dtype=np.uint8) * 245
        cv2.putText(card_img, "FACULTY OF SCIENCE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 100), 2)
        cv2.putText(card_img, "Chaubey Skand", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(card_img, "ID: 1272251121", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(card_img, "Valid Upto: 07/2027", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        _, enc = cv2.imencode(".jpg", card_img)
        resp = client.post(
            "/api/v1/process-id",
            data={"image": (io.BytesIO(enc.tobytes()), "student_badge.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        rec_id = data["id"]

        self.assertEqual(data["extracted_fields"]["full_name"], "Chaubey Skand")
        self.assertEqual(data["extracted_fields"]["id_number"], "1272251121")
        self.assertEqual(data["extracted_fields"]["expiry_date"], "07/2027")
        self.assertIsNone(data["extracted_fields"]["date_of_birth"])

        # Check GET /records/<id>
        get_resp = client.get(f"/api/v1/records/{rec_id}")
        self.assertEqual(get_resp.status_code, 200)
        get_data = get_resp.get_json()
        self.assertEqual(get_data["id"], rec_id)
        self.assertEqual(get_data["extracted_fields"]["full_name"], "Chaubey Skand")

        temp_dir.cleanup()

    def test_leading_noise_in_heuristic_name_cleaned(self):
        """Leading OCR icon noise such as 'ohm' or 'N= ohm' must be cleaned before name."""
        noisy_lines = [
            "GOVT MODEL SENIOR SECONDARY SCHOOL",
            "¢ N= ohm RAHUL KUMAR a",
            "Student ID: STU-2026-001",
            "DOB: 15-08-2010",
        ]
        name = self.extractor._extract_name(noisy_lines, exclude_values=[])
        self.assertEqual(name, "Rahul Kumar", "Leading noise words like 'ohm' must not contaminate the name")

    def test_detector_rejects_non_documents(self):
        """Flat images, blank canvases, and random noise must not be falsely detected as cards."""
        detector = DocumentDetector(min_area_ratio=0.15)

        # 1. Blank white image
        blank_white = np.full((600, 600, 3), 255, dtype=np.uint8)
        _, detected_white = detector.detect_and_align(blank_white)
        self.assertFalse(detected_white, "Blank white image must not be detected as card")

        # 2. Blank black image
        blank_black = np.zeros((600, 600, 3), dtype=np.uint8)
        _, detected_black = detector.detect_and_align(blank_black)
        self.assertFalse(detected_black, "Blank black image must not be detected as card")

        # 3. Random noise
        noise = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)
        _, detected_noise = detector.detect_and_align(noise)
        self.assertFalse(detected_noise, "Random noise image must not be detected as card")

    def test_four_sample_images_regression(self):
        """
        Regression tests against the four sample card images when present:
        - Mit Id College card.jpg
        - image/images.jpg (Rahul)
        - image/images (9).jpg (Jessica)
        - image/images (6).jpg (Estelle)
        """
        detector = DocumentDetector()
        preprocessor = ImagePreprocessor()
        ocr = OCRService()

        sample_cases = [
            {
                "path": r"C:\Users\Skand\OneDrive\Pictures\PersonalImage\PersonalImage\Mit Id College card.jpg",
                "name": "Chaubey Skand",
                "id": "1272251121",
                "dob": None,
                "expiry": "07/2027",
            },
            {
                "path": r"image\images.jpg",
                "name": "Rahul Kumar",
                "id": "STU-2026-001",
                "dob": "15/08/2010",
                "expiry": None,
            },
            {
                "path": r"image\images (9).jpg",
                "name": "Jessica Cook",
                "id": "12345678",
                "dob": None,
                "expiry": None,
            },
            {
                "path": r"image\images (6).jpg",
                "name": "Estelle Darcy",
                "id": "123-456-7890",
                "dob": "12/11/2002",
                "expiry": None,
            },
        ]

        for case in sample_cases:
            if not os.path.isfile(case["path"]):
                continue

            img = cv2.imread(case["path"])
            self.assertIsNotNone(img, f"Failed to read image at {case['path']}")

            warped, detected = detector.detect_and_align(img)
            self.assertTrue(detected, f"Document boundary should be detected for {os.path.basename(case['path'])}")

            pre = preprocessor.preprocess_for_ocr(warped)
            ocr_text = ocr.extract_text(pre)
            fields = self.extractor.extract_fields(ocr_text)

            self.assertEqual(fields["full_name"], case["name"], f"Name mismatch on {os.path.basename(case['path'])}")
            self.assertEqual(fields["id_number"], case["id"], f"ID number mismatch on {os.path.basename(case['path'])}")
            self.assertEqual(fields["date_of_birth"], case["dob"], f"DOB mismatch on {os.path.basename(case['path'])}")
            self.assertEqual(fields["expiry_date"], case["expiry"], f"Expiry mismatch on {os.path.basename(case['path'])}")


if __name__ == "__main__":
    unittest.main()
