"""
ForgeMind Quality Validator.

Scores generated content using embedding-based metrics:
- Semantic Similarity (to original dataset)
- Diversity Score (among generated samples)
- Duplicate Score
- Overall Quality Score

Does NOT use subjective "hallucination scores" in Phase 2.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("sentence-transformers not installed. QualityValidator will be limited.")


@dataclass
class QualityReport:
    """Quality assessment report for a set of generated samples."""

    semantic_similarity: float = 0.0
    diversity_score: float = 0.0
    duplicate_score: float = 0.0
    quality_score: float = 0.0
    total_samples: int = 0
    flagged_low_quality: int = 0


class QualityValidator:
    """
    Embedding-based quality validator for augmented samples.

    Uses sentence-transformers to compute quality metrics
    without relying on subjective heuristics.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None and SentenceTransformer is not None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def validate(
        self,
        generated_samples: list[dict],
        original_samples: list[dict],
        similarity_threshold: float = 0.3,
    ) -> QualityReport:
        """
        Validate the quality of generated samples against originals.

        Args:
            generated_samples: The newly generated samples.
            original_samples: The original dataset samples.
            similarity_threshold: Minimum semantic similarity to originals.

        Returns:
            QualityReport with computed metrics.
        """
        if not generated_samples:
            return QualityReport()

        gen_texts = self._extract_texts(generated_samples)
        orig_texts = self._extract_texts(original_samples)

        if not gen_texts:
            return QualityReport(total_samples=len(generated_samples))

        # Compute embeddings
        if self.model is None:
            logger.warning("SentenceTransformer not available. Returning default scores.")
            return QualityReport(
                total_samples=len(generated_samples),
                quality_score=0.5,
            )

        gen_embeddings = self.model.encode(gen_texts, show_progress_bar=False)
        orig_embeddings = (
            self.model.encode(orig_texts, show_progress_bar=False)
            if orig_texts
            else np.array([])
        )

        # 1. Semantic similarity to originals
        semantic_sim = self._compute_semantic_similarity(gen_embeddings, orig_embeddings)

        # 2. Diversity among generated samples
        diversity = self._compute_diversity(gen_embeddings)

        # 3. Duplicate score (proportion of near-duplicates within generated set)
        dup_score = self._compute_duplicate_score(gen_embeddings)

        # 4. Count low-quality samples
        flagged = 0
        if orig_embeddings.size > 0:
            for emb in gen_embeddings:
                sim = np.dot(emb, orig_embeddings.T).max()
                if sim < similarity_threshold:
                    flagged += 1

        # 5. Overall quality: weighted combination
        quality_score = (
            0.4 * semantic_sim
            + 0.4 * diversity
            + 0.2 * (1.0 - dup_score)
        )

        return QualityReport(
            semantic_similarity=round(semantic_sim, 4),
            diversity_score=round(diversity, 4),
            duplicate_score=round(dup_score, 4),
            quality_score=round(quality_score, 4),
            total_samples=len(generated_samples),
            flagged_low_quality=flagged,
        )

    def _compute_semantic_similarity(
        self, gen_emb: np.ndarray, orig_emb: np.ndarray
    ) -> float:
        """Average max-similarity of each generated sample to the originals."""
        if orig_emb.size == 0 or gen_emb.size == 0:
            return 0.0

        # Normalize embeddings
        gen_norm = gen_emb / (np.linalg.norm(gen_emb, axis=1, keepdims=True) + 1e-9)
        orig_norm = orig_emb / (np.linalg.norm(orig_emb, axis=1, keepdims=True) + 1e-9)

        similarities = np.dot(gen_norm, orig_norm.T)
        max_sims = similarities.max(axis=1)
        return float(max_sims.mean())

    def _compute_diversity(self, embeddings: np.ndarray) -> float:
        """Compute average pairwise dissimilarity among generated samples."""
        if len(embeddings) < 2:
            return 1.0

        norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.dot(norm, norm.T)

        # Get upper triangle (excluding diagonal)
        n = len(sim_matrix)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_sims = sim_matrix[mask]

        if len(pairwise_sims) == 0:
            return 1.0

        # Diversity = 1 - average similarity
        return float(1.0 - pairwise_sims.mean())

    def _compute_duplicate_score(
        self, embeddings: np.ndarray, threshold: float = 0.95
    ) -> float:
        """Proportion of near-duplicate pairs in the generated set."""
        if len(embeddings) < 2:
            return 0.0

        norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.dot(norm, norm.T)

        n = len(sim_matrix)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_sims = sim_matrix[mask]

        duplicates = (pairwise_sims >= threshold).sum()
        total_pairs = len(pairwise_sims)

        return float(duplicates / total_pairs) if total_pairs > 0 else 0.0

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
