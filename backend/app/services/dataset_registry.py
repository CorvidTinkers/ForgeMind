"""
ForgeMind Dataset Registry Service.

Manages dataset versioning and lineage tracking.

Supports:
- Creating initial versions for new datasets
- Incrementing versions for re-uploads / modifications
- Tracking parent-child lineage between dataset versions
"""

import logging
import uuid

from sqlalchemy.orm import Session

from app.domain.models import DatasetVersion

logger = logging.getLogger(__name__)


class DatasetRegistryService:
    """Service for managing dataset versions and lineage."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def register_version(
        self,
        dataset_id: uuid.UUID,
        version: str = "1.0",
        parent_version_id: uuid.UUID | None = None,
        changelog: str | None = None,
    ) -> DatasetVersion:
        """
        Register a new version of a dataset.

        Idempotent: if (dataset_id, version) already exists, returns the
        existing record without raising an error.

        Args:
            dataset_id: The dataset to version.
            version: Semantic version string.
            parent_version_id: ID of the previous version (for lineage).
            changelog: Description of what changed.

        Returns:
            The created (or existing) DatasetVersion record.
        """
        # Check for existing version first to avoid UniqueViolation on retries
        existing = (
            self.db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
            .first()
        )
        if existing is not None:
            logger.info(
                "Version %s for dataset %s already exists, skipping registration",
                version,
                dataset_id,
            )
            return existing

        dataset_version = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            parent_version_id=parent_version_id,
            changelog=changelog,
        )
        self.db.add(dataset_version)
        self.db.commit()
        self.db.refresh(dataset_version)
        logger.info(
            "Registered version %s for dataset %s",
            version,
            dataset_id,
        )
        return dataset_version

    def get_latest_version(self, dataset_id: uuid.UUID) -> DatasetVersion | None:
        """
        Get the latest version of a dataset.

        Args:
            dataset_id: The dataset to query.

        Returns:
            The latest DatasetVersion or None.
        """
        return (
            self.db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.created_at.desc())
            .first()
        )

    def get_version_history(self, dataset_id: uuid.UUID) -> list[DatasetVersion]:
        """
        Get the complete version history of a dataset.

        Args:
            dataset_id: The dataset to query.

        Returns:
            List of DatasetVersion records ordered by creation time.
        """
        return (
            self.db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.created_at.asc())
            .all()
        )

    def get_next_version(self, dataset_id: uuid.UUID) -> str:
        """
        Compute the next version string.

        Args:
            dataset_id: The dataset to query.

        Returns:
            Next version string (e.g., "2.0" if current is "1.0").
        """
        latest = self.get_latest_version(dataset_id)
        if latest is None:
            return "1.0"
        try:
            major = int(float(latest.version)) + 1
            return f"{major}.0"
        except (ValueError, TypeError):
            return "1.0"
