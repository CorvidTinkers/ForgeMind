"""
ForgeMind Domain Detector.

Uses Sentence Transformers for embedding-based semantic classification.
NOT keyword matching.

Approach:
1. Define domain prototypes (descriptions of each domain).
2. Embed dataset samples using a Sentence Transformer model.
3. Embed domain prototypes.
4. Compare using cosine similarity.
5. Return top N domains with confidence scores.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load sentence-transformers to avoid import cost on every request
_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"

# Domain prototypes: rich descriptions for semantic matching.
# Each prototype describes the domain in natural language so that
# cosine similarity produces meaningful results.
DOMAIN_PROTOTYPES: dict[str, str] = {
    "healthcare": (
        "Medical records, clinical notes, patient diagnosis, treatment plans, "
        "pharmaceutical drugs, symptoms, diseases, hospitals, doctors, "
        "healthcare providers, medical imaging, lab results, pathology, "
        "epidemiology, public health, clinical trials."
    ),
    "finance": (
        "Banking transactions, stock market, investment portfolios, "
        "financial statements, credit scores, loans, insurance, "
        "trading, risk assessment, revenue, profit, loss, "
        "cryptocurrency, blockchain, fintech, audit, accounting."
    ),
    "legal": (
        "Legal contracts, court cases, litigation, intellectual property, "
        "compliance, regulations, statutes, jurisprudence, attorneys, "
        "lawsuits, legal precedent, criminal law, civil law, "
        "corporate governance, privacy law, GDPR."
    ),
    "education": (
        "Students, teachers, curriculum, courses, examinations, "
        "grades, schools, universities, online learning, e-learning, "
        "academic research, textbooks, lectures, assignments, "
        "educational technology, pedagogy, tutoring."
    ),
    "customer_support": (
        "Customer complaints, support tickets, help desk, "
        "product issues, refunds, returns, service quality, "
        "chatbot conversations, FAQ, troubleshooting, "
        "customer satisfaction, NPS, feedback, SLA."
    ),
    "technology": (
        "Software development, programming, APIs, databases, "
        "cloud computing, DevOps, machine learning, artificial intelligence, "
        "cybersecurity, networking, system architecture, "
        "web development, mobile apps, data engineering."
    ),
}


def _get_model():
    """Lazy-load the SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Loaded SentenceTransformer model: %s", _MODEL_NAME)
        except Exception as e:
            logger.error("Failed to load SentenceTransformer: %s", e)
    return _model


# Cache domain embeddings (computed once)
_domain_embeddings: dict[str, np.ndarray] = {}


def _get_domain_embeddings() -> dict[str, np.ndarray]:
    """Compute and cache domain prototype embeddings."""
    global _domain_embeddings
    if not _domain_embeddings:
        model = _get_model()
        if model is not None:
            for domain, prototype in DOMAIN_PROTOTYPES.items():
                embedding = model.encode(prototype, normalize_embeddings=True)
                _domain_embeddings[domain] = embedding
            logger.info("Cached %d domain prototype embeddings", len(_domain_embeddings))
    return _domain_embeddings


class DomainDetector:
    """Detects the domain of a dataset using semantic embedding similarity."""

    def detect(
        self,
        sample_texts: list[str],
        top_k: int = 6,
    ) -> dict[str, Any]:
        """
        Classify the dataset domain using embedding-based semantic matching.

        Args:
            sample_texts: List of text samples from the dataset.
            top_k: Number of top domains to return.

        Returns:
            Dict with 'top_domains' list and 'dataset_embedding' array.
        """
        model = _get_model()
        if model is None:
            logger.warning("SentenceTransformer not available. Returning general domain.")
            return {
                "top_domains": [{"domain": "general", "score": 0.5}],
                "dataset_embedding": None,
            }

        domain_embeddings = _get_domain_embeddings()
        if not domain_embeddings:
            return {
                "top_domains": [{"domain": "general", "score": 0.5}],
                "dataset_embedding": None,
            }

        # Sample up to 100 texts for efficiency
        texts_to_embed = sample_texts[:100]
        if not texts_to_embed:
            return {
                "top_domains": [{"domain": "general", "score": 0.5}],
                "dataset_embedding": None,
            }

        # Compute dataset embedding as mean of sample embeddings
        sample_embeddings = model.encode(
            texts_to_embed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dataset_embedding = np.mean(sample_embeddings, axis=0)
        dataset_embedding = dataset_embedding / np.linalg.norm(dataset_embedding)

        # Compute cosine similarity with each domain prototype
        scores = []
        for domain, domain_emb in domain_embeddings.items():
            similarity = float(np.dot(dataset_embedding, domain_emb))
            scores.append({"domain": domain, "score": round(similarity, 4)})

        # Sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_domains = scores[:top_k]

        logger.info(
            "Domain detection results: %s",
            [(d["domain"], d["score"]) for d in top_domains[:3]],
        )

        return {
            "top_domains": top_domains,
            "dataset_embedding": dataset_embedding,
        }
