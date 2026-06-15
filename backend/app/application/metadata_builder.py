"""
ForgeMind Metadata Builder.

Combines outputs from all analyzer submodules and persists
the composite metadata into the Dataset's JSONB column.

Also coordinates:
- Agent Memory storage (embedding + context)
- MLflow tracking (mocked in Phase 1)
- Event publishing
"""

import logging
import uuid
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.domain.enums import DatasetStatus
from app.domain.models import Dataset
from app.infrastructure.embedding_store import EmbeddingStore
from app.infrastructure.mlflow_client import MLflowTrackingService
from app.services.agent_memory import AgentMemoryService

logger = logging.getLogger(__name__)


class MetadataBuilder:
    """
    Builds and persists the final metadata for a dataset.

    Orchestrates the post-analysis workflow:
    1. Store metadata in PostgreSQL JSONB
    2. Store embedding in FAISS via AgentMemory
    3. Log analysis to MLflow (mocked)
    """

    def __init__(
        self,
        db: Session,
        embedding_store: EmbeddingStore,
        mlflow_service: MLflowTrackingService,
    ) -> None:
        self.db = db
        self.agent_memory = AgentMemoryService(db, embedding_store)
        self.mlflow_service = mlflow_service

    def build_and_persist(
        self,
        dataset_id: uuid.UUID,
        metadata: dict[str, Any],
        fingerprint_info: dict[str, Any],
        dataset_embedding: np.ndarray | None = None,
    ) -> Dataset:
        """
        Persist the complete metadata for a dataset.

        Args:
            dataset_id: The dataset to update.
            metadata: Analysis results from the DatasetAnalyzerAgent.
            fingerprint_info: Fingerprint and duplicate detection results.
            dataset_embedding: The embedding vector (for agent memory).

        Returns:
            The updated Dataset record.
        """
        # Merge fingerprint info into metadata
        metadata["fingerprint"] = fingerprint_info.get("fingerprint")
        metadata["is_duplicate"] = fingerprint_info.get("is_duplicate", False)

        # Update the dataset record
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        dataset.metadata_ = metadata
        dataset.fingerprint = fingerprint_info.get("fingerprint")
        dataset.status = DatasetStatus.ANALYZED.value
        self.db.commit()
        self.db.refresh(dataset)

        logger.info("Persisted metadata for dataset %s", dataset_id)

        # Store in agent memory (if embedding available)
        if dataset_embedding is not None:
            context = {
                "domain": (
                    metadata.get("top_domains", [{}])[0].get("domain", "unknown")
                    if metadata.get("top_domains")
                    else "unknown"
                ),
                "task": metadata.get("task_detection", {}).get("task", "unknown"),
                "quality_score": metadata.get("profile", {}).get("quality_score", 0.0),
                "language": metadata.get("language", {}).get("language", "unknown"),
            }
            self.agent_memory.store_memory(
                dataset_id=dataset_id,
                embedding=dataset_embedding,
                context=context,
            )

        # Log to MLflow (mocked)
        self._log_to_mlflow(dataset_id, metadata)

        return dataset

    def _log_to_mlflow(
        self,
        dataset_id: uuid.UUID,
        metadata: dict[str, Any],
    ) -> None:
        """Log analysis results to MLflow tracking service."""
        metrics = {}
        params = {}

        # Extract metrics
        profile = metadata.get("profile", {})
        if profile:
            metrics["quality_score"] = profile.get("quality_score", 0.0)
            metrics["duplication_score"] = profile.get("duplication_score", 0.0)
            metrics["sample_count"] = float(profile.get("sample_count", 0))
            metrics["avg_tokens"] = profile.get("avg_tokens", 0.0)

        # Extract params
        task = metadata.get("task_detection", {})
        if task:
            params["task_type"] = task.get("task", "unknown")
            metrics["task_confidence"] = task.get("confidence", 0.0)

        language = metadata.get("language", {})
        if language:
            params["language"] = language.get("language", "unknown")
            metrics["language_confidence"] = language.get("confidence", 0.0)

        top_domains = metadata.get("top_domains", [])
        if top_domains:
            params["primary_domain"] = top_domains[0].get("domain", "unknown")
            metrics["domain_confidence"] = top_domains[0].get("score", 0.0)

        self.mlflow_service.log_dataset_analysis(
            dataset_id=str(dataset_id),
            metrics=metrics,
            params=params,
            tags={"phase": "1", "component": "dataset_analyzer"},
        )
