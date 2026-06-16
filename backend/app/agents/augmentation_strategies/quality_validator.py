"""
ForgeMind Quality Validator.

Scores generated content using embedding-based metrics:
- Semantic Similarity (to original dataset)
- Diversity Score (among generated samples)
- Duplicate Score
- Overall Quality Score

Phase 2.1: Now also FILTERS samples below the similarity threshold,
returning separate accepted/rejected lists with rejection reasons.
"""

import logging
from dataclasses import dataclass, field

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
    accepted_samples: int = 0
    rejected_samples: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejected_examples: list[dict] = field(default_factory=list)

    # Legacy alias for backward-compatibility
    @property
    def flagged_low_quality(self) -> int:
        return self.rejected_samples


class QualityValidator:
    """
    Embedding-based quality validator for augmented samples.

    Uses sentence-transformers to compute quality metrics.
    Phase 2.1: validates AND filters — rejected samples are removed
    from the output, not just flagged.
    """

    # Maximum rejected examples to include in reports.
    MAX_REJECTED_EXAMPLES = 10

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
    ) -> tuple[list[dict], list[dict], QualityReport]:
        """
        Validate and filter generated samples against originals.

        Args:
            generated_samples: The newly generated samples.
            original_samples: The original dataset samples.
            similarity_threshold: Minimum semantic similarity to originals.

        Returns:
            Tuple of (accepted_samples, rejected_samples, QualityReport).
        """
        if not generated_samples:
            return [], [], QualityReport()

        gen_texts = self._extract_texts(generated_samples)
        orig_texts = self._extract_texts(original_samples)

        if not gen_texts:
            return generated_samples, [], QualityReport(
                total_samples=len(generated_samples),
                accepted_samples=len(generated_samples),
            )

        # --- No model available: pass everything through ---
        if self.model is None:
            logger.warning("SentenceTransformer not available. Returning default scores.")
            return generated_samples, [], QualityReport(
                total_samples=len(generated_samples),
                accepted_samples=len(generated_samples),
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

        # 3. Duplicate score
        dup_score = self._compute_duplicate_score(gen_embeddings)

        # 4. Filter: reject samples below similarity threshold
        accepted = []
        rejected = []
        rejection_reasons: dict[str, int] = {}
        rejected_examples: list[dict] = []

        if orig_embeddings.size > 0:
            gen_norm = gen_embeddings / (
                np.linalg.norm(gen_embeddings, axis=1, keepdims=True) + 1e-9
            )
            orig_norm = orig_embeddings / (
                np.linalg.norm(orig_embeddings, axis=1, keepdims=True) + 1e-9
            )
            sims = np.dot(gen_norm, orig_norm.T)

            for idx, sample in enumerate(generated_samples):
                max_sim = float(sims[idx].max()) if idx < len(sims) else 0.0
                if max_sim < similarity_threshold:
                    rejected.append(sample)
                    reason = "low_semantic_similarity"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    if len(rejected_examples) < self.MAX_REJECTED_EXAMPLES:
                        rejected_examples.append({
                            "sample": {
                                k: v for k, v in sample.items()
                                if isinstance(v, (str, int, float, bool))
                            },
                            "reason": reason,
                            "similarity": round(max_sim, 4),
                        })
                else:
                    accepted.append(sample)
        else:
            accepted = list(generated_samples)

        # 5. Overall quality: weighted combination
        quality_score = (
            0.4 * semantic_sim
            + 0.4 * diversity
            + 0.2 * (1.0 - dup_score)
        )

        report = QualityReport(
            semantic_similarity=round(semantic_sim, 4),
            diversity_score=round(diversity, 4),
            duplicate_score=round(dup_score, 4),
            quality_score=round(quality_score, 4),
            total_samples=len(generated_samples),
            accepted_samples=len(accepted),
            rejected_samples=len(rejected),
            rejection_reasons=rejection_reasons,
            rejected_examples=rejected_examples,
        )

        logger.info(
            "Quality validation: total=%d, accepted=%d, rejected=%d, score=%.4f",
            report.total_samples,
            report.accepted_samples,
            report.rejected_samples,
            report.quality_score,
        )

        return accepted, rejected, report

    def _compute_semantic_similarity(
        self, gen_emb: np.ndarray, orig_emb: np.ndarray
    ) -> float:
        """Average max-similarity of each generated sample to the originals."""
        if orig_emb.size == 0 or gen_emb.size == 0:
            return 0.0

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

        n = len(sim_matrix)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_sims = sim_matrix[mask]

        if len(pairwise_sims) == 0:
            return 1.0

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
