import os
import shutil
import cv2
import numpy as np
import pytesseract
from typing import Optional, Dict, Any


def configure_tesseract_path() -> Optional[str]:
    """Finds and configures the Tesseract binary path."""
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        _set_tessdata_if_present(env_path)
        return env_path

    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        _set_tessdata_if_present(which_path)
        return which_path

    # Standard Windows search candidates
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\tesseract\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Users\Skand\tesseract\tesseract.exe",
    ]

    for cand in candidates:
        if os.path.isfile(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            _set_tessdata_if_present(cand)
            return cand

    return None


def _set_tessdata_if_present(tesseract_exe_path: str) -> None:
    """Sets TESSDATA_PREFIX environment variable if tessdata folder is adjacent."""
    base_dir = os.path.dirname(tesseract_exe_path)
    tessdata_dir = os.path.join(base_dir, "tessdata")
    if os.path.isdir(tessdata_dir) and "TESSDATA_PREFIX" not in os.environ:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir


class OCRService:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.last_passes: Dict[str, str] = {}
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            _set_tessdata_if_present(tesseract_cmd)
        else:
            configure_tesseract_path()

    def run_pass(self, image: np.ndarray, config: str) -> str:
        """Executes a single Tesseract pass safely."""
        try:
            return pytesseract.image_to_string(image, config=config).strip()
        except Exception:
            return ""

    def extract_text(self, preprocessed_images: Dict[str, np.ndarray]) -> str:
        """
        Runs multi-pass OCR across multiple preprocessing representations and PSM modes:
        - Grayscale + PSM 3 (automatic page segmentation baseline)
        - CLAHE + PSM 4 (single column variable size, ideal for cards)
        - CLAHE + PSM 11 (sparse text for floating dates, card IDs, badges)
        - Adaptive threshold + PSM 6 (uniform text blocks)
        - Otsu threshold + PSM 6
        Preserves individual passes in self.last_passes for debugging.
        Intelligently merges unique, informative lines prioritizing high-confidence passes.
        """
        self.last_passes = {}

        # 1. Grayscale baseline with PSM 3
        if "grayscale" in preprocessed_images:
            self.last_passes["grayscale_psm3"] = self.run_pass(
                preprocessed_images["grayscale"], "--oem 3 --psm 3"
            )

        # 2. CLAHE enhanced with PSM 3, PSM 4, PSM 6, and PSM 11
        if "clahe" in preprocessed_images:
            self.last_passes["clahe_psm3"] = self.run_pass(
                preprocessed_images["clahe"], "--oem 3 --psm 3"
            )
            self.last_passes["clahe_psm4"] = self.run_pass(
                preprocessed_images["clahe"], "--oem 3 --psm 4"
            )
            self.last_passes["clahe_psm6"] = self.run_pass(
                preprocessed_images["clahe"], "--oem 3 --psm 6"
            )
            self.last_passes["clahe_psm11"] = self.run_pass(
                preprocessed_images["clahe"], "--oem 3 --psm 11"
            )

        # 3. Adaptive threshold with PSM 6
        if "adaptive_threshold" in preprocessed_images:
            self.last_passes["adaptive_psm6"] = self.run_pass(
                preprocessed_images["adaptive_threshold"], "--oem 3 --psm 6"
            )

        # 4. Otsu threshold with PSM 6
        if "otsu_threshold" in preprocessed_images:
            self.last_passes["otsu_psm6"] = self.run_pass(
                preprocessed_images["otsu_threshold"], "--oem 3 --psm 6"
            )

        # Priority order for merging: Grayscale -> CLAHE -> Thresholds
        pass_priority = [
            "grayscale_psm3",
            "clahe_psm3",
            "clahe_psm4",
            "clahe_psm6",
            "clahe_psm11",
            "adaptive_psm6",
            "otsu_psm6",
        ]
        merged_lines = []
        normalized_seen = set()

        for pass_name in pass_priority:
            text = self.last_passes.get(pass_name, "")
            if not text:
                continue
            for line in text.splitlines():
                cleaned = line.strip()
                # Skip empty or 1-char noise
                if not cleaned or len(cleaned) < 2:
                    continue
                # Skip lines that contain only non-alphanumeric noise symbols
                if not any(c.isalnum() for c in cleaned):
                    continue

                # Normalized alphanumeric key for deduplication
                norm_key = "".join(c.lower() for c in cleaned if c.isalnum())
                if not norm_key or len(norm_key) < 2:
                    continue

                # Check if this line is already covered or can replace a shorter truncated line
                is_duplicate = False
                replaced = False
                for idx, existing_line in enumerate(merged_lines):
                    existing_key = "".join(c.lower() for c in existing_line if c.isalnum())
                    if norm_key == existing_key:
                        is_duplicate = True
                        break

                    # Only upgrade numeric/code strings (e.g. 272251121 -> 1272251121)
                    # Never overwrite clean alphabetic name lines with noisy lower-priority lines
                    is_numeric_upgrade = (
                        any(c.isdigit() for c in existing_key)
                        and any(c.isdigit() for c in norm_key)
                        and not any(c.isalpha() for c in existing_key)
                        and not any(c.isalpha() for c in norm_key)
                    )
                    if is_numeric_upgrade and len(existing_key) >= 5 and existing_key in norm_key and len(norm_key) > len(existing_key):
                        merged_lines[idx] = cleaned
                        normalized_seen.discard(existing_key)
                        normalized_seen.add(norm_key)
                        replaced = True
                        break

                    # If an existing line already contains this line, or if existing text covers it, skip
                    if len(norm_key) >= 6 and norm_key in existing_key and len(existing_key) >= len(norm_key):
                        is_duplicate = True
                        break
                    if len(existing_key) >= 6 and existing_key in norm_key:
                        is_duplicate = True
                        break

                if not is_duplicate and not replaced:
                    normalized_seen.add(norm_key)
                    merged_lines.append(cleaned)

        return "\n".join(merged_lines)


