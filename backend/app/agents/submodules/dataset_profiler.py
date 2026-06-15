"""
ForgeMind Dataset Profiler.

Generates statistical profile of a dataset:
- Sample count
- Token statistics (avg, max, min)
- Language distribution
- Duplication score
- Quality score
"""

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


class DatasetProfiler:
    """Generates statistical profiles for datasets."""

    def profile(
        self,
        sample_texts: list[str],
        language_distribution: dict[str, float] | None = None,
    ) -> dict:
        """
        Generate a comprehensive dataset profile.

        Args:
            sample_texts: List of text samples from the dataset.
            language_distribution: Pre-computed language distribution (optional).

        Returns:
            Profile dict with statistical metrics.
        """
        if not sample_texts:
            return {
                "sample_count": 0,
                "avg_tokens": 0.0,
                "max_tokens": 0,
                "min_tokens": 0,
                "language_distribution": language_distribution or {},
                "duplication_score": 0.0,
                "quality_score": 0.0,
            }

        # Token counts (whitespace tokenization for speed)
        token_counts = [len(self._tokenize(text)) for text in sample_texts]

        sample_count = len(sample_texts)
        avg_tokens = round(sum(token_counts) / sample_count, 2)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)

        # Duplication score
        duplication_score = self._compute_duplication(sample_texts)

        # Quality score (composite metric)
        quality_score = self._compute_quality(
            sample_texts=sample_texts,
            token_counts=token_counts,
            duplication_score=duplication_score,
        )

        profile = {
            "sample_count": sample_count,
            "avg_tokens": avg_tokens,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "language_distribution": language_distribution or {},
            "duplication_score": round(duplication_score, 4),
            "quality_score": round(quality_score, 4),
        }

        logger.info(
            "Dataset profile: samples=%d, avg_tokens=%.1f, quality=%.2f",
            sample_count,
            avg_tokens,
            quality_score,
        )
        return profile

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer."""
        return text.split()

    def _compute_duplication(self, texts: list[str]) -> float:
        """
        Compute the ratio of duplicate texts.

        Uses exact string matching. A more sophisticated approach
        could use MinHash or SimHash for near-duplicate detection.
        """
        if len(texts) <= 1:
            return 0.0

        # Normalize texts for comparison
        normalized = [t.strip().lower() for t in texts]
        unique_count = len(set(normalized))
        total_count = len(normalized)

        return 1.0 - (unique_count / total_count)

    def _compute_quality(
        self,
        sample_texts: list[str],
        token_counts: list[int],
        duplication_score: float,
    ) -> float:
        """
        Compute a composite quality score (0.0 - 1.0).

        Factors:
        - Content length adequacy
        - Token distribution consistency
        - Duplication ratio (penalty)
        - Empty/short sample ratio (penalty)
        """
        if not sample_texts:
            return 0.0

        total = len(sample_texts)

        # Factor 1: Content adequacy (penalize very short samples)
        short_threshold = 5  # tokens
        short_count = sum(1 for tc in token_counts if tc < short_threshold)
        content_score = 1.0 - (short_count / total)

        # Factor 2: Token consistency (low variance is better for structured data)
        if len(token_counts) > 1:
            mean_tc = sum(token_counts) / len(token_counts)
            variance = sum((tc - mean_tc) ** 2 for tc in token_counts) / len(token_counts)
            std_dev = variance ** 0.5
            cv = std_dev / mean_tc if mean_tc > 0 else 0
            consistency_score = max(0.0, 1.0 - (cv / 3.0))
        else:
            consistency_score = 0.5

        # Factor 3: Duplication penalty
        duplication_penalty = 1.0 - duplication_score

        # Factor 4: Empty sample penalty
        empty_count = sum(1 for t in sample_texts if not t.strip())
        empty_score = 1.0 - (empty_count / total)

        # Weighted combination
        quality = (
            content_score * 0.30
            + consistency_score * 0.20
            + duplication_penalty * 0.30
            + empty_score * 0.20
        )

        return min(max(quality, 0.0), 1.0)
