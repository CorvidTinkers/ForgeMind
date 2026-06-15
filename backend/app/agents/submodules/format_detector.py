"""
ForgeMind Format Detector.

Detects the file format (PDF, TXT, CSV, JSON) with a confidence score.
Uses magic bytes and structural analysis rather than just file extension.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Magic bytes for common file formats
MAGIC_BYTES = {
    "pdf": b"%PDF",
}


class FormatDetector:
    """Detects file format using magic bytes and structural heuristics."""

    def detect(self, data: bytes, filename: str) -> dict:
        """
        Detect file format with confidence score.

        Args:
            data: Raw file bytes.
            filename: Original filename.

        Returns:
            Dict with 'format' and 'confidence' keys.
        """
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Check if the file is binary garbage
        if self._is_binary(data):
            # Check if PDF magic bytes match
            if data[:4] == MAGIC_BYTES.get("pdf", b""):
                return {"format": "pdf", "confidence": 0.99}
            
            # If binary but has a known extension, lower confidence
            if extension in ("pdf", "txt", "csv", "json"):
                return {"format": extension, "confidence": 0.30}
            return {"format": "unknown", "confidence": 0.0}

        # Check magic bytes first
        if data[:4] == MAGIC_BYTES.get("pdf", b""):
            return {"format": "pdf", "confidence": 0.99}

        # Try JSON parsing
        if self._is_json(data):
            confidence = 0.95 if extension == "json" else 0.80
            return {"format": "json", "confidence": confidence}

        # Try CSV detection
        if self._is_csv(data):
            confidence = 0.95 if extension == "csv" else 0.75
            return {"format": "csv", "confidence": confidence}

        # Fall back to TXT
        if self._is_text(data):
            confidence = 0.90 if extension == "txt" else 0.70
            return {"format": "txt", "confidence": confidence}

        # Use extension as last resort
        if extension in ("pdf", "txt", "csv", "json"):
            return {"format": extension, "confidence": 0.50}

        return {"format": "unknown", "confidence": 0.0}

    def _is_binary(self, data: bytes) -> bool:
        """Check if data contains binary content."""
        if not data:
            return False
        # Null bytes are a clear indicator of binary data
        if b"\x00" in data:
            return True
        # Check percentage of control/non-printable characters
        control_chars = sum(
            1 for b in data if (b < 32 and b not in (9, 10, 13)) or b == 127
        )
        if (control_chars / len(data)) > 0.30:
            return True
        return False


    def _is_json(self, data: bytes) -> bool:
        """Check if data is valid JSON."""
        try:
            text = data.decode("utf-8", errors="strict")
            json.loads(text)
            return True
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False

    def _is_csv(self, data: bytes) -> bool:
        """Check if data looks like CSV (has consistent delimiters)."""
        try:
            text = data.decode("utf-8", errors="replace")
            lines = text.strip().split("\n")[:10]
            if len(lines) < 2:
                return False

            # Check for consistent comma/tab/pipe counts
            for delimiter in [",", "\t", "|", ";"]:
                counts = [line.count(delimiter) for line in lines]
                if counts[0] > 0 and len(set(counts)) <= 2:
                    return True
            return False
        except Exception:
            return False

    def _is_text(self, data: bytes) -> bool:
        """Check if data is valid UTF-8 text."""
        try:
            data.decode("utf-8", errors="strict")
            return True
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ["latin-1", "utf-16"]:
                try:
                    data.decode(encoding, errors="strict")
                    return True
                except (UnicodeDecodeError, Exception):
                    continue
            return False
