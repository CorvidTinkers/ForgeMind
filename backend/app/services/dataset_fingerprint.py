"""
ForgeMind Dataset Fingerprinting Engine.

Computes SHA256 hashes for uploaded datasets to enable:
- Duplicate detection
- Dataset versioning
- Future lineage tracking
"""

import hashlib
import logging
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.domain.models import Dataset

logger = logging.getLogger(__name__)

# Read files in 64KB chunks to handle large files without memory issues.
CHUNK_SIZE = 65536


class DatasetFingerprintService:
    """Service for computing and managing dataset fingerprints."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def compute_fingerprint(self, data: BinaryIO) -> str:
        """
        Compute SHA256 hash of a file stream.

        Args:
            data: A seekable binary file stream.

        Returns:
            Hex-encoded SHA256 hash string.
        """
        sha256 = hashlib.sha256()
        data.seek(0)
        while chunk := data.read(CHUNK_SIZE):
            sha256.update(chunk)
        data.seek(0)  # Reset stream position for subsequent reads
        fingerprint = sha256.hexdigest()
        logger.info("Computed fingerprint: %s", fingerprint[:16] + "...")
        return fingerprint

    def check_duplicate(self, fingerprint: str) -> dict:
        """
        Check if a dataset with the same fingerprint already exists.

        Args:
            fingerprint: SHA256 hash to check.

        Returns:
            Dict with 'is_duplicate' flag and optional 'existing_dataset_id'.
        """
        existing = (
            self.db.query(Dataset)
            .filter(Dataset.fingerprint == fingerprint)
            .first()
        )
        if existing:
            logger.info(
                "Duplicate detected: fingerprint=%s, existing_id=%s",
                fingerprint[:16],
                existing.id,
            )
            return {
                "is_duplicate": True,
                "existing_dataset_id": str(existing.id),
            }
        return {"is_duplicate": False, "existing_dataset_id": None}

    def fingerprint_and_check(self, data: BinaryIO) -> dict:
        """
        Compute fingerprint and check for duplicates in one call.

        Args:
            data: A seekable binary file stream.

        Returns:
            Dict with 'fingerprint', 'is_duplicate', and optional 'existing_dataset_id'.
        """
        fingerprint = self.compute_fingerprint(data)
        duplicate_info = self.check_duplicate(fingerprint)
        return {
            "fingerprint": fingerprint,
            **duplicate_info,
        }
