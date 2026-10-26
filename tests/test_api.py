import io
import os
import tempfile
import unittest
import numpy as np
import cv2
from PIL import Image, ImageDraw

from app import create_app


def create_synthetic_id_image_bytes(text_lines=None) -> bytes:
    """Generates an image of an ID card in bytes."""
    width, height = 800, 500
    img = Image.new("RGB", (width, height), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)

    # Card border
    draw.rectangle([(10, 10), (width - 10, height - 10)], outline=(50, 50, 80), width=4)
    # Header bar
    draw.rectangle([(12, 12), (width - 12, 80)], fill=(30, 60, 120))
    draw.text((250, 35), "STATE IDENTIFICATION CARD", fill=(255, 255, 255))

    lines = text_lines or [
        "NAME: MICHAEL SCOTT",
        "ID NUMBER: DL-78901234",
        "DATE OF BIRTH: 15/03/1965",
        "EXPIRY DATE: 20/05/2030",
        "SEX: M  CLASS: C",
    ]

    y = 120
    for line in lines:
        draw.text((50, y), line, fill=(20, 20, 20))
        y += 55

    # Paste face photo if asset exists
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "sample_face.jpg")
    if os.path.exists(asset_path):
        face_pil = Image.open(asset_path).convert("RGB")
        face_pil = face_pil.resize((200, 240))
        img.paste(face_pil, (550, 120))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class TestIDProcessingAPI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_api.db")
        self.thumbnails_dir = os.path.join(self.temp_dir.name, "thumbnails")

        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": self.db_path,
            "THUMBNAIL_DIR": self.thumbnails_dir,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_post_missing_file_returns_400(self):
        response = self.client.post("/api/v1/process-id", data={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_post_unsupported_file_extension_returns_400(self):
        data = {
            "image": (io.BytesIO(b"fake pdf content"), "document.pdf")
        }
        response = self.client.post(
            "/api/v1/process-id",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.get_json()["error"])

    def test_post_corrupted_image_returns_400(self):
        data = {
            "image": (io.BytesIO(b"Not an image at all"), "broken.jpg")
        }
        response = self.client.post(
            "/api/v1/process-id",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unreadable or corrupted", response.get_json()["error"])

    def test_post_valid_id_and_get_record(self):
        img_bytes = create_synthetic_id_image_bytes()
        data = {
            "image": (io.BytesIO(img_bytes), "sample_dl.jpg")
        }
        post_resp = self.client.post(
            "/api/v1/process-id",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(post_resp.status_code, 201)
        res_data = post_resp.get_json()

        # Validate response structure
        self.assertIn("id", res_data)
        record_id = res_data["id"]
        self.assertTrue(len(record_id) > 10)

        self.assertIn("metadata", res_data)
        meta = res_data["metadata"]
        self.assertEqual(meta["filename"], "sample_dl.jpg")
        self.assertEqual(meta["width"], 800)
        self.assertEqual(meta["height"], 500)
        self.assertEqual(meta["channels"], 3)
        self.assertEqual(meta["format"], "JPG")
        self.assertIn("file_size_bytes", meta)
        self.assertIn("created_at", meta)

        self.assertIn("extracted_fields", res_data)
        fields = res_data["extracted_fields"]
        self.assertIn("full_name", fields)
        self.assertIn("id_number", fields)
        self.assertIn("date_of_birth", fields)
        self.assertIn("expiry_date", fields)

        # Assert face thumbnail extraction
        self.assertIsNotNone(res_data.get("face_thumbnail_path"))
        thumb_path = res_data["face_thumbnail_path"]
        full_thumb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), thumb_path)
        self.assertTrue(os.path.exists(full_thumb_path), f"Thumbnail file should exist at {full_thumb_path}")

        # Now retrieve via GET /api/v1/records/{id}
        get_resp = self.client.get(f"/api/v1/records/{record_id}")
        self.assertEqual(get_resp.status_code, 200)
        get_data = get_resp.get_json()
        self.assertEqual(get_data["id"], record_id)
        self.assertEqual(get_data["metadata"]["filename"], "sample_dl.jpg")
        self.assertEqual(get_data["extracted_fields"], fields)
        self.assertEqual(get_data["face_thumbnail_path"], thumb_path)

    def test_get_non_existent_record_returns_404(self):
        response = self.client.get("/api/v1/records/non-existent-uuid-999")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
