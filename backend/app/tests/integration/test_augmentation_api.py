"""
ForgeMind Integration Tests — Augmentation API Endpoints.

Tests the augmentation API endpoints using TestClient.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.domain.enums import AugmentationStatus, DatasetStatus


# ============================================
# Test fixtures
# ============================================

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    return db


@pytest.fixture
def client(mock_db):
    """Create a test client with mocked dependencies."""
    from app.api.dependencies import get_db

    with patch("app.main.init_db"):
        from app.main import app

        def override_get_db():
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


# ============================================
# Augment Endpoint Tests
# ============================================

class TestAugmentEndpoint:
    """Tests for POST /api/v1/datasets/{id}/augment."""

    @patch("app.api.augmentation.run_augmentation_task")
    def test_augment_success(self, mock_task, client, mock_db):
        fake_id = uuid.uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = fake_id
        mock_dataset.status = DatasetStatus.ANALYZED.value

        # Dataset query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        # No active augmentation run (second query)
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_dataset,  # dataset query
            None,          # active run query
        ]

        def mock_add(obj):
            if hasattr(obj, "id") and obj.id is None:
                obj.id = uuid.uuid4()
                
        mock_db.add.side_effect = mock_add

        response = client.post(f"/api/v1/datasets/{fake_id}/augment")
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"

    def test_augment_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        fake_id = str(uuid.uuid4())
        response = client.post(f"/api/v1/datasets/{fake_id}/augment")
        assert response.status_code == 404

    def test_augment_not_analyzed(self, client, mock_db):
        fake_id = uuid.uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = fake_id
        mock_dataset.status = DatasetStatus.UPLOADED.value

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        response = client.post(f"/api/v1/datasets/{fake_id}/augment")
        assert response.status_code == 409


# ============================================
# Augmentation Run Status Tests
# ============================================

class TestAugmentationRunEndpoint:
    """Tests for GET /api/v1/augmentation/{run_id}."""

    def test_get_run_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/augmentation/{fake_id}")
        assert response.status_code == 404

    def test_get_run_success(self, client, mock_db):
        from datetime import datetime, timezone

        fake_id = uuid.uuid4()
        mock_run = MagicMock()
        mock_run.id = fake_id
        mock_run.dataset_id = uuid.uuid4()
        mock_run.status = AugmentationStatus.COMPLETED.value
        mock_run.config = {}
        mock_run.report = {"final_samples": 100}
        mock_run.created_at = datetime.now(timezone.utc)
        mock_run.completed_at = datetime.now(timezone.utc)

        mock_db.query.return_value.filter.return_value.first.return_value = mock_run

        response = client.get(f"/api/v1/augmentation/{fake_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


# ============================================
# Augmentation Report Tests
# ============================================

class TestAugmentationReportEndpoint:
    """Tests for GET /api/v1/augmentation/{run_id}/report."""

    def test_report_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/augmentation/{fake_id}/report")
        assert response.status_code == 404

    def test_report_not_completed(self, client, mock_db):
        fake_id = uuid.uuid4()
        mock_run = MagicMock()
        mock_run.id = fake_id
        mock_run.status = AugmentationStatus.RUNNING.value

        mock_db.query.return_value.filter.return_value.first.return_value = mock_run

        response = client.get(f"/api/v1/augmentation/{fake_id}/report")
        assert response.status_code == 409
