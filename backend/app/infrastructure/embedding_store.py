"""
ForgeMind Embedding Store.

FAISS-based vector store for persisting dataset embeddings.
These embeddings are generated during domain detection and stored
for reuse by future agents (Data Augmentation, Pipeline Planner, etc.).
"""

import logging
import os
import pickle
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# FAISS is imported lazily to allow the module to load even if
# faiss-cpu is not yet installed (e.g., during testing).
try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not installed. Embedding store will be unavailable.")


class EmbeddingStore:
    """
    FAISS-based vector store for dataset embeddings.

    Stores embeddings alongside metadata so that future agents
    can perform similarity searches across historical datasets.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: str = "/app/data/faiss_store",
    ) -> None:
        """
        Initialize the embedding store.

        Args:
            dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2).
            index_path: Directory to persist the FAISS index.
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "index.faiss")
        self.metadata_file = os.path.join(index_path, "metadata.pkl")
        self.metadata_store: list[dict[str, Any]] = []
        self.index = None

        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing index from disk or create a new one."""
        if faiss is None:
            logger.warning("FAISS not available. Skipping index creation.")
            return

        os.makedirs(self.index_path, exist_ok=True)

        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata_store = pickle.load(f)
            logger.info(
                "Loaded FAISS index with %d vectors", self.index.ntotal
            )
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = []
            logger.info("Created new FAISS index (dim=%d)", self.dimension)

    def add(
        self,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> int:
        """
        Add an embedding with associated metadata to the store.

        Args:
            embedding: A 1D numpy array of shape (dimension,).
            metadata: Dictionary of metadata to associate with the vector.

        Returns:
            The index ID of the added vector.
        """
        if self.index is None:
            logger.warning("FAISS index not available. Skipping add.")
            return -1

        # Normalize for cosine similarity (IndexFlatIP on normalized = cosine)
        embedding = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)

        idx = self.index.ntotal
        self.index.add(embedding)
        self.metadata_store.append(metadata)
        self._persist()
        return idx

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for the most similar embeddings.

        Args:
            query_embedding: A 1D numpy array of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'score' and 'metadata' keys.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata_store):
                results.append({
                    "score": float(score),
                    "metadata": self.metadata_store[idx],
                })
        return results

    def _persist(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata_store, f)


def get_embedding_store() -> EmbeddingStore:
    """Factory function for dependency injection."""
    return EmbeddingStore()
