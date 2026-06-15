"""
ForgeMind Pydantic V2 Schemas.

Request/response schemas for the API layer. Separated from SQLAlchemy
models to maintain clean architecture boundaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================
# Upload Schemas
# ============================================

class UploadResponse(BaseModel):
    """Response returned after a successful dataset upload."""

    dataset_id: uuid.UUID
    status: str = "uploaded"


# ============================================
# Analysis Schemas
# ============================================

class AnalyzeRequest(BaseModel):
    """Optional configuration for analysis (future extensibility)."""

    force_reanalysis: bool = False


class AnalyzeResponse(BaseModel):
    """Response returned after triggering analysis."""

    dataset_id: uuid.UUID
    status: str
    message: str = "Analysis task submitted"


# ============================================
# Status Schemas
# ============================================

class StatusResponse(BaseModel):
    """Response for dataset processing status."""

    dataset_id: uuid.UUID
    status: str
    validation_errors: list[dict[str, Any]] | None = None


# ============================================
# Metadata / Intelligence Schemas
# ============================================

class DomainScore(BaseModel):
    """A single domain classification result with confidence."""

    domain: str
    score: float


class FormatDetectionResult(BaseModel):
    """Output from the format detector."""

    format: str
    confidence: float


class LanguageDetectionResult(BaseModel):
    """Output from the language detector."""

    language: str
    confidence: float


class TaskDetectionResult(BaseModel):
    """Output from the task/intent detector."""

    task: str
    confidence: float


class DatasetProfile(BaseModel):
    """Statistical profile of the dataset."""

    sample_count: int = 0
    avg_tokens: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    language_distribution: dict[str, float] = Field(default_factory=dict)
    duplication_score: float = 0.0
    quality_score: float = 0.0


class EntityRelationship(BaseModel):
    """A relationship between two extracted entities."""

    source: str
    target: str
    relation: str


class KnowledgeExtractionResult(BaseModel):
    """Output from the knowledge extractor."""

    entities: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    relationships: list[EntityRelationship] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    """
    The complete dataset intelligence report.

    Combines outputs from all analyzer submodules into a single
    structured representation stored as JSONB in PostgreSQL.
    """

    format_detection: FormatDetectionResult | None = None
    language: LanguageDetectionResult | None = None
    top_domains: list[DomainScore] = Field(default_factory=list)
    task_detection: TaskDetectionResult | None = None
    profile: DatasetProfile | None = None
    knowledge: KnowledgeExtractionResult | None = None
    fingerprint: str | None = None
    is_duplicate: bool = False


# ============================================
# Dataset Full Response
# ============================================

class DatasetVersionInfo(BaseModel):
    """Version information from the registry."""

    version: str
    parent_version_id: uuid.UUID | None = None
    changelog: str | None = None
    created_at: datetime


class DatasetResponse(BaseModel):
    """Full dataset intelligence report returned by GET /datasets/{id}."""

    id: uuid.UUID
    filename: str
    original_filename: str
    storage_path: str
    fingerprint: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    status: str
    metadata: DatasetMetadata | None = None
    validation_errors: list[dict[str, Any]] | None = None
    versions: list[DatasetVersionInfo] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================
# Phase 2: Augmentation Schemas
# ============================================

class AugmentRequest(BaseModel):
    """Request body for triggering augmentation."""

    target_multiplier: int | None = Field(
        default=None,
        description="Optional override for target sample multiplier.",
    )
    strategies: list[str] | None = Field(
        default=None,
        description="Optional override for augmentation strategies.",
    )


class AugmentResponse(BaseModel):
    """Response after triggering an augmentation run."""

    run_id: uuid.UUID
    dataset_id: uuid.UUID
    status: str = "pending"
    message: str = "Augmentation task submitted"


class AugmentationRunResponse(BaseModel):
    """Response for GET /augmentation/{run_id}."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    config: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    created_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class AugmentationReportResponse(BaseModel):
    """Response for GET /augmentation/{run_id}/report."""

    run_id: uuid.UUID
    original_samples: int = 0
    generated_samples: int = 0
    duplicates_removed: int = 0
    final_samples: int = 0
    quality_score: float = 0.0
    strategies_used: list[dict[str, Any]] = Field(default_factory=list)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    deduplication_report: dict[str, Any] = Field(default_factory=dict)
    audit_trails: list[dict[str, Any]] = Field(default_factory=list)


class TrainingDatasetResponse(BaseModel):
    """Response for GET /training-dataset/{id}."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    augmentation_run_id: uuid.UUID | None = None
    format_name: str
    record_count: int
    created_at: datetime

    model_config = {"from_attributes": True}

