# Document AI - ID Card Processing Service

A Flask backend service that processes ID card images, extracts information (Name, ID number, DOB, Expiry Date), crops the face portrait, and stores the records in SQLite.

## Features
- Document detection and perspective alignment using OpenCV
- Image preprocessing (grayscale, CLAHE, adaptive thresholding)
- Multi-pass OCR using Tesseract (PSM 3, 4, 6, 11)
- Rule-based and regex field extraction (Name, ID, DOB, Expiry)
- Face detection and portrait cropping using Haar Cascade
- REST API with SQLite storage

## Tech Stack
- Python 3.9+
- Flask
- OpenCV (`opencv-python`)
- Pytesseract
- SQLite

## Setup Instructions

### 1. Install Tesseract OCR
- **Windows:** Download and install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Make sure `tesseract.exe` is added to your PATH or set `TESSERACT_CMD`.
- **Linux:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`

### 2. Install Python Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python app.py
```
The server will start at `http://127.0.0.1:5000`.

## API Endpoints

### 1. Process ID Card
- **URL:** `POST /api/v1/process-id`
- **Method:** `POST`
- **Body:** `multipart/form-data` with field `image`

**Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/v1/process-id -F "image=@sample_id.jpg"
```

**Response (201 Created):**
```json
{
  "id": "e0a6d71b-32db-4fc8-99e5-9c84918e93fd",
  "metadata": {
    "filename": "sample_id.jpg",
    "width": 1280,
    "height": 800,
    "channels": 3,
    "format": "JPG",
    "file_size_bytes": 145820,
    "document_detected": true,
    "created_at": "2026-10-26T06:00:00+00:00"
  },
  "extracted_fields": {
    "full_name": "Rahul Kumar",
    "id_number": "STU-2026-001",
    "date_of_birth": "15/08/2010",
    "expiry_date": null
  },
  "face_thumbnail_path": "storage/thumbnails/e0a6d71b-32db-4fc8-99e5-9c84918e93fd_face.jpg",
  "raw_text": "..."
}
```

### 2. Get Record by ID
- **URL:** `GET /api/v1/records/<id>`

**Example:**
```bash
curl http://127.0.0.1:5000/api/v1/records/e0a6d71b-32db-4fc8-99e5-9c84918e93fd
```

### 3. Health Check
- **URL:** `GET /health`

## Running Tests
Run all unit and integration tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
