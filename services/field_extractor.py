import re
from typing import Dict, Optional, Any, List

DATE_PATTERNS = [
    r"\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})\b",             # DD/MM/YYYY or MM/DD/YYYY
    r"\b(\d{4})[\/\-\.](\d{2})[\/\-\.](\d{2})\b",             # YYYY-MM-DD
    r"\b(\d{2})[\/\-\.](\d{4})\b",                             # MM/YYYY (common for card validity)
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4})\b",  # 15 Jan 1990
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[\s,]+\d{4})\b", # Jan 15, 1990
]

PHONE_KEYWORDS = {"emergency", "phone", "mobile", "cell", "tel", "contact", "call", "fax"}

DOC_HEADER_BLACKLIST = {
    "DRIVER", "LICENSE", "DRIVING", "IDENTIFICATION", "CARD", "REPUBLIC", "STATE",
    "UNITED", "STATES", "GOVERNMENT", "DEPARTMENT", "VEHICLES", "MINISTRY", "EMPLOYEE",
    "IDENTITY", "OFFICIAL", "NATIONAL", "AUTHORITY", "BADGE", "FEDERATION", "PROVINCE",
    "SEX", "CLASS", "REST", "END", "TYPE", "DATE", "BIRTH", "EXPIRY", "EXPIRES", "HEIGHT",
    "EYES", "HAIR", "FACULTY", "COLLEGE", "UNIVERSITY", "INSTITUTE", "ENGINEERING",
    "TECHNOLOGY", "FEELNOLOGY", "MIT", "WPU", "MIT-WPU", "MCA", "BCA", "BTECH", "MTECH",
    "MBA", "PHD", "EMERGENCY", "VALID", "UPTO", "GROUP", "STUDENT", "STAFF", "PERMANENT",
    "ACCOUNT", "INCOME", "TAX", "INDIA", "GOVT", "SIGNATURE", "FATHER", "FATHERS",
    "BLOOD", "ADDRESS", "ISSUE", "ISSUED", "EXP", "NUMBER", "OLOGY", "OLOGIES", "FEE",
    "ACADEMY", "CAMPUS", "MANAGEMENT", "COUNCIL", "BOARD", "SCHOOL", "SECONDARY",
    "SENIOR", "PRIMARY", "HIGHER", "EDUCATION", "PUBLIC"
}



class FieldExtractor:
    def extract_fields(self, raw_text: str) -> Dict[str, Optional[str]]:
        """
        Parses raw OCR text to extract:
        - full_name
        - id_number
        - date_of_birth
        - expiry_date
        Returns None for any field that cannot be reliably extracted.
        """
        if not raw_text or not raw_text.strip():
            return {
                "full_name": None,
                "id_number": None,
                "date_of_birth": None,
                "expiry_date": None,
            }

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        dob = self._extract_dob(raw_text, lines)
        expiry = self._extract_expiry(raw_text, lines)
        id_number = self._extract_id_number(raw_text, lines)
        name = self._extract_name(lines, exclude_values=[dob, expiry, id_number])

        return {
            "full_name": name,
            "id_number": id_number,
            "date_of_birth": dob,
            "expiry_date": expiry,
        }

    def _extract_dob(self, text: str, lines: List[str]) -> Optional[str]:
        """Extracts date of birth based on explicit labels."""
        dob_patterns = [
            r"(?:D\.?O\.?B\.?|Date\s+of\s+Birth|Birth\s+Date|Born\s*On|Born)\s*[:;\-\s]*([0-9A-Za-z\/\-\.\s\)]{4,20})",
            r"(?:DOB)\s*[:;\-\s]*([0-9\/\-\.]{8,10})",
        ]
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                parsed = self._find_first_date(val)
                if parsed:
                    return parsed

        # Check for multi-line format where date is on next line after DOB label
        for i, line in enumerate(lines):
            if re.search(r"^(?:D\.?O\.?B\.?|Date\s+of\s+Birth|Birth\s+Date|Born)\s*[:;\-]?$", line, re.IGNORECASE):
                if i + 1 < len(lines):
                    parsed = self._find_first_date(lines[i + 1])
                    if parsed:
                        return parsed
            elif re.search(r"\b(?:DOB|BIRTH|BORN)\b", line, re.IGNORECASE):
                parsed = self._find_first_date(line)
                if parsed:
                    return parsed

        return None

    def _extract_expiry(self, text: str, lines: List[str]) -> Optional[str]:
        """Extracts expiry date based on explicit labels."""
        exp_patterns = [
            r"(?:EXP(?:IRY|IRATION)?(?:\s*DATE)?|EXPIRES|VALID\s*(?:UP\s*TO|UPTO|UNTIL|THRU|TO|TILL))\s*[:;\-\s]*([0-9A-Za-z\/\-\.\s\)]{4,20})",
            r"(?:EXP)\s*[:;\-\s]*([0-9\/\-\.]{4,10})",
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                parsed = self._find_first_date(val)
                if parsed:
                    return parsed

        # Multi-line format
        for i, line in enumerate(lines):
            if re.search(r"^(?:EXP(?:IRY|IRATION)?(?:\s*DATE)?|VALID\s*(?:UP\s*TO|UPTO|UNTIL|THRU|TO))\s*[:;\-]?$", line, re.IGNORECASE):
                if i + 1 < len(lines):
                    parsed = self._find_first_date(lines[i + 1])
                    if parsed:
                        return parsed
            elif re.search(r"\b(?:EXP|EXPIRY|EXPIRES|VALID\s*UPTO|VALID\s*UP\s*TO|VALID\s*THRU|VALID\s*UNTIL)\b", line, re.IGNORECASE):
                parsed = self._find_first_date(line)
                if parsed:
                    return parsed

        return None

    def _find_first_date(self, text: str) -> Optional[str]:
        """Finds and normalizes the first valid date string."""
        if not text:
            return None

        # Clean common OCR trailing punctuation
        cleaned = text.strip(" \t\n\r,;:)")

        # Tolerant DD/MM/YYYY pattern with minor delimiter distortion like "06/03) 2004"
        loose_match = re.search(r"\b(\d{2})[\/\-\.\s\)]+(\d{2})[\/\-\.\s]+(\d{4})\b", cleaned)
        if loose_match:
            d, m, y = loose_match.groups()
            return f"{d}/{m}/{y}"

        # Standard date patterns
        for pattern in DATE_PATTERNS:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                return m.group(0).strip(" \t\n\r,;:)")

        return None

    def _extract_id_number(self, text: str, lines: List[str]) -> Optional[str]:
        """
        Extracts ID number using explicit labels, format patterns, and context checks
        to avoid confusing IDs with phone numbers or dates.
        """
        # 1. Explicit labeled patterns
        id_label_patterns = [
            r"(?:ID\s*(?:NO|NUMBER)?|DL\s*(?:NO|NUMBER)?|Roll\s*(?:NO|NUMBER)?|PRN|Reg(?:istration)?\s*(?:NO|NUMBER)?|Enrollment\s*(?:NO|NUMBER)?|PAN|Card\s*(?:NO|NUMBER)?|Lic(?:ense)?\s*(?:NO|NUMBER)?|Doc(?:ument)?\s*(?:NO|NUMBER)?|Badge\s*(?:NO|NUMBER)?)\s*[:\-#.]*\s*([A-Z0-9\-]{5,22})\b",
            r"(?:Permanent\s*Account\s*Number(?:\s*Card)?)\s*[:\-#.]*\s*([A-Z0-9]{10})\b",
        ]
        for pattern in id_label_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate = match.group(1).strip()
                if self._is_valid_id(candidate):
                    return candidate

        # 2. Check each line for labels and tokens
        for i, line in enumerate(lines):
            # Exclude lines that mention emergency/phone numbers
            if any(k in line.lower() for k in PHONE_KEYWORDS):
                continue

            if re.search(r"\b(?:ID|LIC|NO|NUM|DL|PRN|ROLL|REG|PAN)\b", line, re.IGNORECASE):
                tokens = re.findall(r"[A-Z0-9\-]{5,20}", line)
                for token in tokens:
                    if self._is_valid_id(token):
                        return token

                # If label is alone on this line, check next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not any(k in next_line.lower() for k in PHONE_KEYWORDS):
                        tokens = re.findall(r"[A-Z0-9\-]{5,20}", next_line)
                        for token in tokens:
                            if self._is_valid_id(token):
                                return token

        # 3. Standard ID formats without labels
        # Indian PAN format (5 letters, 4 digits, 1 letter)
        pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
        if pan_match:
            return pan_match.group(1)

        # Standard Driver License format (e.g., D12345678 or DL-1420110012345)
        dl_match = re.search(r"\b([A-Z]{1,2}[0-9]{7,12})\b", text)
        if dl_match and self._is_valid_id(dl_match.group(1)):
            return dl_match.group(1)

        # Hyphenated ID format (e.g. EMP-1234-5678)
        hyph_match = re.search(r"\b([A-Z0-9]{3,5}-[A-Z0-9]{3,6}-[A-Z0-9]{3,6})\b", text)
        if hyph_match and self._is_valid_id(hyph_match.group(1)):
            return hyph_match.group(1)

        # Pure numeric ID (8-14 digits, e.g. student ID, PRN)
        numeric_candidates = []
        for line in lines:
            if any(k in line.lower() for k in PHONE_KEYWORDS):
                continue
            numeric_tokens = re.findall(r"\b(\d{8,14})\b", line)
            for num in numeric_tokens:
                # Avoid standard 10-digit mobile numbers starting with 6, 7, 8, 9 unless labeled
                if len(num) == 10 and num[0] in "6789":
                    continue
                if self._is_valid_id(num):
                    numeric_candidates.append(num)

        if numeric_candidates:
            return max(numeric_candidates, key=len)

        return None

    def _is_valid_id(self, val: str) -> bool:
        if not val or len(val) < 5 or len(val) > 22:
            return False
        # Must contain at least one digit
        if not re.search(r"\d", val):
            return False
        # Shouldn't be a pure date
        if self._find_first_date(val) == val:
            return False
        # Avoid common phone numbers (e.g. 12 digits starting with 91)
        if len(val) == 12 and val.startswith("91") and val[2] in "6789":
            return False
        return True

    def _extract_name(self, lines: List[str], exclude_values: List[Optional[str]]) -> Optional[str]:
        cleaned_excludes = {v.lower() for v in exclude_values if v}

        # 1. Look for explicit Name: or LN / FN prefix
        name_patterns = [
            r"(?:Full\s*Name|Student\s*Name|Cardholder\s*Name|Holder\s*Name|Name|Holder|Given\s*Name|Surname)\s*[:\-]\s*([A-Za-z\s\.\,\'\-]+)",
            r"(?:LN|FN)\s*[:\-]\s*([A-Za-z\s\.\,\'\-]+)",
        ]
        for line in lines:
            for pat in name_patterns:
                m = re.search(pat, line, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    if self._is_valid_name(cand, cleaned_excludes):
                        return self._format_name(cand)

        # 2. Heuristic line search: Find uppercase/titlecase text line that represents a personal name
        for line in lines:
            # Skip if contains numbers, phone/emergency keywords, or excluded values
            if re.search(r"\d", line):
                continue
            if any(k in line.lower() for k in PHONE_KEYWORDS):
                continue
            if any(exc in line.lower() for exc in cleaned_excludes):
                continue

            # Strip non-alpha characters and discard single-char noise tokens (e.g. 'i', 'l', 'e')
            raw_words = re.sub(r"[^A-Za-z\s\,\'\.]", " ", line).split()
            valid_words = [w.strip(".,'") for w in raw_words if len(w.strip(".,'")) >= 2]

            # Strip leading lowercase noise tokens before capitalized/uppercase name tokens (e.g. 'ohm' before 'RAHUL KUMAR')
            while len(valid_words) >= 3 and valid_words[0].islower() and any(w.isupper() or w.istitle() for w in valid_words[1:]):
                valid_words.pop(0)

            # Strip trailing short lowercase noise tokens
            while len(valid_words) >= 3 and valid_words[-1].islower() and len(valid_words[-1]) <= 2:
                valid_words.pop(-1)

            cleaned = " ".join(valid_words).strip()
            if self._is_valid_name(cleaned, cleaned_excludes):
                return self._format_name(cleaned)


        return None

    def _is_valid_name(self, name: str, excludes: set) -> bool:
        if not name or len(name) < 4 or len(name) > 40:
            return False

        words = name.split()
        if len(words) < 2 or len(words) > 4:
            return False

        # Check each word: must not be blacklisted and must have length >= 3
        for word in words:
            clean_word = word.strip(".,'").upper()
            if clean_word in DOC_HEADER_BLACKLIST:
                return False
            if len(clean_word) < 3:
                return False
            if not clean_word.isalpha():
                return False

        if name.lower() in excludes:
            return False

        return True

    def _format_name(self, name: str) -> str:
        """Standardizes name formatting."""
        cleaned = name.strip(" \t\n\r,.:;-")
        # Handle "SURNAME, GIVEN NAME" -> "GIVEN NAME SURNAME"
        if "," in cleaned:
            parts = [p.strip(" \t\n\r,.:;-") for p in cleaned.split(",") if p.strip(" \t\n\r,.:;-")]
            if len(parts) == 2:
                return f"{parts[1]} {parts[0]}".title()
        return cleaned.title()

