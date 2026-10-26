import os
import sqlite3
from typing import Optional

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "records.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    target_path = db_path or os.environ.get("DATABASE_PATH", DEFAULT_DB_PATH)
    db_dir = os.path.dirname(os.path.abspath(target_path))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS id_records (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    image_width INTEGER,
                    image_height INTEGER,
                    image_channels INTEGER,
                    image_format TEXT,
                    file_size_bytes INTEGER,
                    document_detected INTEGER DEFAULT 0,
                    full_name TEXT,
                    id_number TEXT,
                    date_of_birth TEXT,
                    expiry_date TEXT,
                    face_thumbnail_path TEXT,
                    raw_text TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
    finally:
        conn.close()
