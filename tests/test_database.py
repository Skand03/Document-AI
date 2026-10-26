import os
import tempfile
import unittest
from database.db import init_db
from database.record_repository import RecordRepository


class TestDatabaseAndRepository(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        init_db(self.db_path)
        self.repo = RecordRepository(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_record(self):
        record_data = {
            "id": "rec-12345",
            "filename": "sample_id.jpg",
            "image_width": 800,
            "image_height": 500,
            "image_channels": 3,
            "image_format": "JPG",
            "file_size_bytes": 10240,
            "document_detected": True,
            "full_name": "Jane Smith",
            "id_number": "D9876543",
            "date_of_birth": "1992-05-14",
            "expiry_date": "2028-05-14",
            "face_thumbnail_path": "storage/thumbnails/rec-12345_face.jpg",
            "raw_text": "STATE DRIVER LICENSE\nName: Jane Smith\nDOB: 1992-05-14\nEXP: 2028-05-14\nDL: D9876543",
            "created_at": "2026-10-26T12:00:00Z",
        }

        saved = self.repo.save(record_data)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["id"], "rec-12345")
        self.assertEqual(saved["extracted_fields"]["full_name"], "Jane Smith")
        self.assertEqual(saved["extracted_fields"]["id_number"], "D9876543")
        self.assertEqual(saved["extracted_fields"]["date_of_birth"], "1992-05-14")
        self.assertEqual(saved["extracted_fields"]["expiry_date"], "2028-05-14")
        self.assertTrue(saved["metadata"]["document_detected"])
        self.assertEqual(saved["metadata"]["filename"], "sample_id.jpg")

        retrieved = self.repo.get_by_id("rec-12345")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], "rec-12345")
        self.assertEqual(retrieved["extracted_fields"]["full_name"], "Jane Smith")

    def test_get_non_existent_record_returns_none(self):
        result = self.repo.get_by_id("does-not-exist")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
