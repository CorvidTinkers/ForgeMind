"""
ForgeMind Language Detector.

Uses FastText for robust language identification with confidence scores.
Falls back to a simple heuristic if FastText model is not available.
"""

import logging
import os
import re
import tempfile
from collections import Counter

logger = logging.getLogger(__name__)

# Attempt to load FastText
try:
    import fasttext

    # Suppress FastText warnings about loading model
    fasttext.FastText.eprint = lambda x: None
    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False
    logger.warning("FastText not installed. Using fallback language detection.")

_model = None
_MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH", "/app/models/lid.176.ftz")


def _get_model():
    """Lazy-load the FastText language identification model."""
    global _model
    if _model is None and _FASTTEXT_AVAILABLE:
        if os.path.exists(_MODEL_PATH):
            _model = fasttext.load_model(_MODEL_PATH)
            logger.info("Loaded FastText model from %s", _MODEL_PATH)
        else:
            logger.warning(
                "FastText model not found at %s. "
                "Download lid.176.ftz from fasttext.cc/docs/en/language-identification.html",
                _MODEL_PATH,
            )
    return _model


class LanguageDetector:
    """Detects the primary language of text content."""

    def detect(self, text: str) -> dict:
        """
        Detect language of the provided text.

        Args:
            text: Text content to analyze.

        Returns:
            Dict with 'language' and 'confidence' keys.
        """
        if not text or not text.strip():
            return {"language": "unknown", "confidence": 0.0}

        # Clean text for detection
        cleaned = self._clean_text(text)
        if len(cleaned) < 10:
            return {"language": "unknown", "confidence": 0.0}

        model = _get_model()
        if model is not None:
            return self._detect_fasttext(model, cleaned)
        return self._detect_fallback(cleaned)

    def detect_distribution(self, texts: list[str]) -> dict[str, float]:
        """
        Detect language distribution across multiple text samples.

        Args:
            texts: List of text samples.

        Returns:
            Dict mapping language codes to their proportion.
        """
        if not texts:
            return {}

        lang_counts = Counter()
        for text in texts:
            result = self.detect(text)
            if result["confidence"] > 0.5:
                lang_counts[result["language"]] += 1

        total = sum(lang_counts.values())
        if total == 0:
            return {"unknown": 1.0}

        return {
            lang: round(count / total, 3)
            for lang, count in lang_counts.most_common()
        }

    def _detect_fasttext(self, model, text: str) -> dict:
        """Detect language using FastText model."""
        # FastText expects single-line input
        text_clean = text.replace("\n", " ")[:5000]
        predictions = model.predict(text_clean, k=1)
        label = predictions[0][0].replace("__label__", "")
        confidence = float(predictions[1][0])
        return {"language": label, "confidence": round(confidence, 4)}

    def _detect_fallback(self, text: str) -> dict:
        """
        Simple heuristic-based fallback for language detection.
        Checks character ranges for common languages.
        """
        # Count character types
        ascii_count = sum(1 for c in text if ord(c) < 128)
        total = len(text)

        if total == 0:
            return {"language": "unknown", "confidence": 0.0}

        ascii_ratio = ascii_count / total

        if ascii_ratio > 0.9:
            return {"language": "en", "confidence": 0.60}
        elif any("\u4e00" <= c <= "\u9fff" for c in text):
            return {"language": "zh", "confidence": 0.70}
        elif any("\u0900" <= c <= "\u097f" for c in text):
            return {"language": "hi", "confidence": 0.70}
        elif any("\u0600" <= c <= "\u06ff" for c in text):
            return {"language": "ar", "confidence": 0.70}
        else:
            return {"language": "en", "confidence": 0.40}

    def _clean_text(self, text: str) -> str:
        """Remove URLs, emails, and excessive whitespace."""
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
