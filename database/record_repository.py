import sqlite3
from typing import Optional, Dict, Any
from database.db import get_db_connection


class RecordRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        return get_db_connection(self.db_path)

    def save(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persists a new record into SQLite."""
        conn = self._get_connection()
        query = """
            INSERT INTO id_records (
                id, filename, image_width, image_height, image_channels,
                image_format, file_size_bytes, document_detected,
                full_name, id_number, date_of_birth, expiry_date,
                face_thumbnail_path, raw_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            record_data["id"],
            record_data.get("filename"),
            record_data.get("image_width"),
            record_data.get("image_height"),
            record_data.get("image_channels"),
            record_data.get("image_format"),
            record_data.get("file_size_bytes"),
            1 if record_data.get("document_detected") else 0,
            record_data.get("full_name"),
            record_data.get("id_number"),
            record_data.get("date_of_birth"),
            record_data.get("expiry_date"),
            record_data.get("face_thumbnail_path"),
            record_data.get("raw_text"),
            record_data.get("created_at"),
        )
        try:
            with conn:
                conn.execute(query, params)
            return self.get_by_id(record_data["id"])  # Return formatted saved record
        finally:
            conn.close()

    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a record by its unique ID."""
        conn = self._get_connection()
        query = "SELECT * FROM id_records WHERE id = ?"
        try:
            cursor = conn.cursor()
            cursor.execute(query, (record_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "id": row["id"],
                "metadata": {
                    "filename": row["filename"],
                    "width": row["image_width"],
                    "height": row["image_height"],
                    "channels": row["image_channels"],
                    "format": row["image_format"],
                    "file_size_bytes": row["file_size_bytes"],
                    "document_detected": bool(row["document_detected"]),
                    "created_at": row["created_at"],
                },
                "extracted_fields": {
                    "full_name": row["full_name"],
                    "id_number": row["id_number"],
                    "date_of_birth": row["date_of_birth"],
                    "expiry_date": row["expiry_date"],
                },
                "face_thumbnail_path": row["face_thumbnail_path"],
                "raw_text": row["raw_text"],
            }
        finally:
            conn.close()
