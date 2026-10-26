import os
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from services.document_detector import DocumentDetector
from services.image_preprocessor import ImagePreprocessor
from services.ocr_service import OCRService
from services.field_extractor import FieldExtractor
from services.face_extractor import FaceExtractor
from database.record_repository import RecordRepository

id_blueprint = Blueprint("id_routes", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@id_blueprint.route("/api/v1/process-id", methods=["POST"])
def process_id():
    """
    Accepts an ID-card image upload, validates, detects document boundary,
    aligns ROI, preprocesses for OCR, runs OCR, extracts fields,
    extracts face thumbnail, persists record in SQLite, and returns structured JSON.
    """
    # 1. Validate file presence
    file = None
    if "image" in request.files:
        file = request.files["image"]
    elif "file" in request.files:
        file = request.files["file"]

    if file is None or file.filename == "":
        return jsonify({"error": "No image file provided in request. Use 'image' or 'file' form field."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Supported formats: JPG, JPEG, PNG, BMP, WEBP, TIFF."}), 400

    try:
        # Read raw bytes
        raw_bytes = file.read()
        if len(raw_bytes) == 0:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        # Decode image using OpenCV
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            return jsonify({"error": "Image file is unreadable or corrupted."}), 400

        # 2. Extract image metadata
        safe_name = secure_filename(file.filename) or "uploaded_image.jpg"
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) == 3 else 1
        fmt = safe_name.rsplit(".", 1)[-1].upper() if "." in safe_name else "UNKNOWN"
        file_size = len(raw_bytes)
        record_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # 3. Document boundary detection & alignment
        detector = DocumentDetector()
        aligned_image, doc_detected = detector.detect_and_align(image)

        # 4. Portrait/face thumbnail extraction
        thumbnail_dir = current_app.config.get("THUMBNAIL_DIR")
        face_extractor = FaceExtractor(output_dir=thumbnail_dir)
        face_thumbnail_path = face_extractor.extract_and_save(aligned_image, record_id)

        # If no face found in aligned card, try original image as fallback
        if face_thumbnail_path is None and doc_detected:
            face_thumbnail_path = face_extractor.extract_and_save(image, record_id)

        # 5. Image preprocessing for OCR
        preprocessor = ImagePreprocessor()
        preprocessed_dict = preprocessor.preprocess_for_ocr(aligned_image)

        # 6. OCR execution
        ocr_service = OCRService()
        raw_text = ocr_service.extract_text(preprocessed_dict)

        # 7. Field extraction
        field_extractor = FieldExtractor()
        extracted_fields = field_extractor.extract_fields(raw_text)

        # 8. Persist to SQLite
        repo = RecordRepository(db_path=current_app.config.get("DATABASE_PATH"))
        record_payload = {
            "id": record_id,
            "filename": safe_name,
            "image_width": w,
            "image_height": h,
            "image_channels": c,
            "image_format": fmt,
            "file_size_bytes": file_size,
            "document_detected": doc_detected,
            "full_name": extracted_fields["full_name"],
            "id_number": extracted_fields["id_number"],
            "date_of_birth": extracted_fields["date_of_birth"],
            "expiry_date": extracted_fields["expiry_date"],
            "face_thumbnail_path": face_thumbnail_path,
            "raw_text": raw_text,
            "created_at": created_at,
        }

        saved_record = repo.save(record_payload)
        return jsonify(saved_record), 201

    except Exception:
        # Avoid leaking internal stack trace to clients
        return jsonify({"error": "An internal error occurred while processing the ID card."}), 500


@id_blueprint.route("/api/v1/records/<record_id>", methods=["GET"])
def get_record(record_id: str):
    """Retrieves a previously processed ID record by its unique ID."""
    if not record_id or not record_id.strip():
        return jsonify({"error": "Invalid record ID."}), 400

    try:
        repo = RecordRepository(db_path=current_app.config.get("DATABASE_PATH"))
        record = repo.get_by_id(record_id.strip())

        if record is None:
            return jsonify({"error": f"Record with ID '{record_id}' not found."}), 404

        return jsonify(record), 200

    except Exception:
        return jsonify({"error": "An error occurred while retrieving the record."}), 500
