"""
ForgeMind Agent Memory Service.

Stores dataset embeddings and contextual metadata so that future
agents (Pipeline Planner, Evaluation Agent) can learn from
historical analyses.

Memory entries link a dataset to:
- Its embedding vector (stored in FAISS)
- Contextual metadata (domain, task, recommended strategies)
"""

import logging
import uuid
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.domain.models import AgentMemory
from app.infrastructure.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


class AgentMemoryService:
    """Service for storing and retrieving agent memory."""

    def __init__(
        self,
        db: Session,
        embedding_store: EmbeddingStore,
    ) -> None:
        self.db = db
        self.embedding_store = embedding_store

    def store_memory(
        self,
        dataset_id: uuid.UUID,
        embedding: np.ndarray,
        context: dict[str, Any],
    ) -> AgentMemory:
        """
        Store a dataset embedding and context in agent memory.

        Args:
            dataset_id: The dataset this memory belongs to.
            embedding: The embedding vector (from SentenceTransformers).
            context: Contextual metadata (domain, task, scores, etc.).

        Returns:
            The created AgentMemory record.
        """
        # Store embedding in FAISS
        faiss_id = self.embedding_store.add(
            embedding=embedding,
            metadata={
                "dataset_id": str(dataset_id),
                **context,
            },
        )

        # Store reference in PostgreSQL
        memory = AgentMemory(
            dataset_id=dataset_id,
            embedding_id=str(faiss_id),
            context=context,
        )
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)

        logger.info(
            "Stored agent memory for dataset %s (faiss_id=%d)",
            dataset_id,
            faiss_id,
        )
        return memory

    def find_similar_datasets(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find datasets with similar embeddings.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of similar datasets to return.

        Returns:
            List of similar dataset memories with scores.
        """
        return self.embedding_store.search(query_embedding, top_k=top_k)

    def get_memory_for_dataset(
        self,
        dataset_id: uuid.UUID,
    ) -> AgentMemory | None:
        """
        Retrieve the memory entry for a specific dataset.

        Args:
            dataset_id: The dataset to look up.

        Returns:
            AgentMemory record or None.
        """
        return (
            self.db.query(AgentMemory)
            .filter(AgentMemory.dataset_id == dataset_id)
            .first()
        )
