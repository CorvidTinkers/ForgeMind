"""
ForgeMind Domain Models (SQLAlchemy 2.0).

Defines the core database entities:
- Dataset: The primary entity with JSONB metadata.
- DatasetVersion: Registry for dataset versioning and lineage.
- AgentMemory: Stores embeddings and contextual memory for future planners.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ForgeMind models."""
    pass


class Dataset(Base):
    """
    Primary dataset entity.

    Stores file metadata and analysis results in a JSONB column
    to avoid rigid schema migrations as the analysis evolves.
    """

    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    storage_path = Column(String(1024), nullable=False, comment="MinIO object path")
    fingerprint = Column(String(64), nullable=True, comment="SHA256 hash")
    file_size = Column(Integer, nullable=True, comment="File size in bytes")
    mime_type = Column(String(128), nullable=True)
    status = Column(
        String(32),
        nullable=False,
        default="uploaded",
        index=True,
    )
    metadata_ = Column(
        "metadata",
        JSONB,
        nullable=True,
        default=dict,
        comment="Analysis results: domain, task, language, profile, knowledge",
    )
    validation_errors = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Validation issues found during pre-analysis check",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    versions = relationship(
        "DatasetVersion",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, filename={self.filename}, status={self.status})>"


class DatasetVersion(Base):
    """
    Dataset Registry for versioning and lineage tracking.

    Supports tracking dataset evolution:
    Medical Dataset V1 → V2 → V3
    """

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version = Column(String(32), nullable=False, default="1.0")
    parent_version_id = Column(
        UUID(as_uuid=True),
        ForeignKey("dataset_versions.id", ondelete="SET NULL"),
        nullable=True,
        comment="Points to the previous version for lineage",
    )
    changelog = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    dataset = relationship("Dataset", back_populates="versions")
    parent_version = relationship("DatasetVersion", remote_side=[id])

    def __repr__(self) -> str:
        return f"<DatasetVersion(dataset_id={self.dataset_id}, version={self.version})>"


class AgentMemory(Base):
    """
    Agent Memory for storing dataset intelligence that future agents can query.

    Stores the embedding vector reference and contextual metadata so that
    the Pipeline Planner can learn from historical dataset analyses.
    """

    __tablename__ = "agent_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding_id = Column(
        String(256),
        nullable=True,
        comment="Reference to the FAISS vector index ID",
    )
    context = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Contextual metadata: domain, task, recommended_strategy, etc.",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<AgentMemory(dataset_id={self.dataset_id})>"


# ============================================
# Phase 2: Data Augmentation Models
# ============================================

class AugmentationRun(Base):
    """
    Tracks a single augmentation run.

    Stores the configuration, status, and generation audit trail
    for full traceability and reproducibility.
    """

    __tablename__ = "augmentation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
    )
    config = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Augmentation plan: strategies, targets, ratios",
    )
    audit_trail = Column(
        JSONB,
        nullable=True,
        default=list,
        comment="Generation audit: generator, prompt_template, temperature, counts",
    )
    report = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment="Final augmentation report with quality metrics",
    )
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    augmented_datasets = relationship(
        "AugmentedDataset",
        back_populates="augmentation_run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<AugmentationRun(id={self.id}, dataset_id={self.dataset_id}, status={self.status})>"


class AugmentedDataset(Base):
    """
    Stores the augmented samples produced by an augmentation run.
    """

    __tablename__ = "augmented_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    augmentation_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("augmentation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    samples = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="The augmented samples as a JSON array",
    )
    sample_count = Column(Integer, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    augmentation_run = relationship("AugmentationRun", back_populates="augmented_datasets")

    def __repr__(self) -> str:
        return f"<AugmentedDataset(id={self.id}, samples={self.sample_count})>"


class TrainingDataset(Base):
    """
    Stores the final compiled training dataset in a standard format
    (Alpaca, ChatML) in the Training Dataset Repository.
    """

    __tablename__ = "training_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    augmentation_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("augmentation_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    format_name = Column(
        String(32),
        nullable=False,
        default="alpaca",
        comment="Output format: alpaca, chatml",
    )
    records = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="The training records in the specified format",
    )
    record_count = Column(Integer, nullable=False, default=0)
    storage_path = Column(
        String(1024),
        nullable=True,
        comment="Optional MinIO path for large datasets",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<TrainingDataset(id={self.id}, format={self.format_name}, records={self.record_count})>"

