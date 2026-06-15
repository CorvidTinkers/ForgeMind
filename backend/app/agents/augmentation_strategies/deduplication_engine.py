"""
ForgeMind Deduplication Engine.

Reuses Phase 1 FAISS embeddings to detect and remove highly similar
(near-duplicate) samples from the augmented dataset. Compares generated
samples against both the original dataset and other generated samples.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning(
        "sentence-transformers not installed. DeduplicationEngine will be limited."
    )


@dataclass
class DeduplicationReport:
    """Report from a deduplication run."""

    total_input: int = 0
    duplicates_removed: int = 0
    final_count: int = 0
    duplicate_pairs: list[tuple[int, int, float]] | None = None


class DeduplicationEngine:
    """
    Embedding-based deduplication engine.

    Uses cosine similarity between sentence embeddings to detect
    near-duplicate samples that should be removed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None and SentenceTransformer is not None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def deduplicate(
        self,
        generated_samples: list[dict],
        original_samples: list[dict] | None = None,
    ) -> tuple[list[dict], DeduplicationReport]:
        """
        Remove near-duplicate samples.

        Checks for duplicates:
        1. Among the generated samples themselves.
        2. Between generated samples and the original dataset.

        Args:
            generated_samples: The augmented samples to deduplicate.
            original_samples: Original dataset samples to compare against.

        Returns:
            Tuple of (deduplicated samples, deduplication report).
        """
        if not generated_samples:
            return [], DeduplicationReport()

        total_input = len(generated_samples)

        # Extract texts
        gen_texts = self._extract_texts(generated_samples)
        orig_texts = self._extract_texts(original_samples or [])

        if not gen_texts or self.model is None:
            logger.warning("Cannot deduplicate: missing model or texts.")
            return generated_samples, DeduplicationReport(
                total_input=total_input,
                final_count=total_input,
            )

        # Compute embeddings
        gen_embeddings = self.model.encode(gen_texts, show_progress_bar=False)
        gen_norm = gen_embeddings / (
            np.linalg.norm(gen_embeddings, axis=1, keepdims=True) + 1e-9
        )

        # Track indices to remove
        remove_indices = set()

        # 1. Check against original samples
        if orig_texts:
            orig_embeddings = self.model.encode(orig_texts, show_progress_bar=False)
            orig_norm = orig_embeddings / (
                np.linalg.norm(orig_embeddings, axis=1, keepdims=True) + 1e-9
            )

            sim_to_orig = np.dot(gen_norm, orig_norm.T)
            for i in range(len(gen_norm)):
                if sim_to_orig[i].max() >= self.similarity_threshold:
                    remove_indices.add(i)

        # 2. Check within generated samples
        sim_matrix = np.dot(gen_norm, gen_norm.T)
        for i in range(len(gen_norm)):
            if i in remove_indices:
                continue
            for j in range(i + 1, len(gen_norm)):
                if j in remove_indices:
                    continue
                if sim_matrix[i][j] >= self.similarity_threshold:
                    # Remove the later one
                    remove_indices.add(j)

        # Filter out duplicates
        deduplicated = [
            sample
            for idx, sample in enumerate(generated_samples)
            if idx not in remove_indices
        ]

        report = DeduplicationReport(
            total_input=total_input,
            duplicates_removed=len(remove_indices),
            final_count=len(deduplicated),
        )

        logger.info(
            "Deduplication: input=%d, removed=%d, final=%d",
            total_input,
            len(remove_indices),
            len(deduplicated),
        )

        return deduplicated, report

    def _extract_texts(self, samples: list[dict]) -> list[str]:
        """Extract text content from sample dicts."""
        texts = []
        text_keys = ["text", "content", "question", "instruction", "input"]
        for sample in samples:
            for key in text_keys:
                if key in sample and isinstance(sample[key], str) and sample[key].strip():
                    texts.append(sample[key].strip())
                    break
            else:
                parts = [
                    str(v).strip()
                    for v in sample.values()
                    if isinstance(v, str) and v.strip()
                ]
                if parts:
                    texts.append(" ".join(parts))
        return texts
