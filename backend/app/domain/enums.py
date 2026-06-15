"""
ForgeMind Domain Enums.

Defines all status codes, dataset types, and domain categories
used throughout the application. These enums are the single source
of truth for categorical values.
"""

import enum


class DatasetStatus(str, enum.Enum):
    """Lifecycle status of a dataset."""

    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"
    FINGERPRINTING = "fingerprinting"
    FINGERPRINTED = "fingerprinted"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    ANALYSIS_FAILED = "analysis_failed"


class DatasetType(str, enum.Enum):
    """Supported dataset task types."""

    QA = "qa"
    CONVERSATION = "conversation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    RAW_DOCUMENTS = "raw_documents"
    UNKNOWN = "unknown"


class DatasetDomain(str, enum.Enum):
    """Supported domain categories for semantic classification."""

    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    LEGAL = "legal"
    EDUCATION = "education"
    CUSTOMER_SUPPORT = "customer_support"
    TECHNOLOGY = "technology"
    GENERAL = "general"


class FileFormat(str, enum.Enum):
    """Supported file formats for upload."""

    PDF = "pdf"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    UNKNOWN = "unknown"


# ============================================
# Phase 2: Data Augmentation Enums
# ============================================

class AugmentationStatus(str, enum.Enum):
    """Lifecycle status of an augmentation run."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    VALIDATING = "validating"
    DEDUPLICATING = "deduplicating"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"


class AugmentationStrategyType(str, enum.Enum):
    """Available augmentation strategies."""

    PARAPHRASE = "paraphrase"
    SYNTHETIC_QA = "synthetic_qa"
    SYNTHETIC_SAMPLE = "synthetic_sample"
    CONVERSATION = "conversation"


class TrainingFormat(str, enum.Enum):
    """Supported training dataset output formats."""

    ALPACA = "alpaca"
    CHATML = "chatml"

