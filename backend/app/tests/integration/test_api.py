"""
ForgeMind Integration Tests — API Endpoints.

Tests the FastAPI endpoints using TestClient.
Requires database and service mocking for isolation.
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.domain.enums import DatasetStatus
from app.domain.models import Base, Dataset


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

    with patch("app.main.init_db") as mock_init_db:
        from app.main import app

        def override_get_db():
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()



# ============================================
# Upload Endpoint Tests
# ============================================

class TestUploadEndpoint:
    """Tests for POST /api/v1/datasets/upload."""

    @patch("app.api.datasets.get_minio_client")
    @patch("app.api.datasets.get_event_publisher")
    def test_upload_csv_success(self, mock_publisher, mock_minio, client, mock_db):
        # Setup mocks
        mock_minio_instance = MagicMock()
        mock_minio_instance.upload_file.return_value = "forgemind-datasets/datasets/test/data.csv"
        mock_minio.return_value = mock_minio_instance

        mock_pub_instance = MagicMock()
        mock_publisher.return_value = mock_pub_instance

        # Create test file
        csv_content = b"question,answer\nWhat is AI?,Artificial Intelligence\n"

        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", csv_content, "text/csv")},
        )

        assert response.status_code == 201
        data = response.json()
        assert "dataset_id" in data
        assert data["status"] == "uploaded"

    def test_upload_unsupported_format(self, client):
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.docx", b"content", "application/docx")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_empty_file(self, client):
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", b"", "text/csv")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


# ============================================
# Status Endpoint Tests
# ============================================

class TestStatusEndpoint:
    """Tests for GET /api/v1/datasets/{id}/status."""

    def test_get_status_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/datasets/{fake_id}/status")
        assert response.status_code == 404

    def test_get_status_success(self, client, mock_db):
        fake_id = uuid.uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = fake_id
        mock_dataset.status = DatasetStatus.ANALYZED.value
        mock_dataset.validation_errors = []

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        response = client.get(f"/api/v1/datasets/{fake_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "analyzed"


# ============================================
# Analyze Endpoint Tests
# ============================================

class TestAnalyzeEndpoint:
    """Tests for POST /api/v1/datasets/analyze/{id}."""

    @patch("app.api.datasets.analyze_dataset_task")
    def test_analyze_success(self, mock_task, client, mock_db):
        fake_id = uuid.uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = fake_id
        mock_dataset.status = DatasetStatus.UPLOADED.value

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        response = client.post(f"/api/v1/datasets/analyze/{fake_id}")
        assert response.status_code == 200
        mock_task.delay.assert_called_once_with(str(fake_id))

    def test_analyze_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        fake_id = str(uuid.uuid4())
        response = client.post(f"/api/v1/datasets/analyze/{fake_id}")
        assert response.status_code == 404

    def test_analyze_already_processing(self, client, mock_db):
        fake_id = uuid.uuid4()
        mock_dataset = MagicMock()
        mock_dataset.id = fake_id
        mock_dataset.status = DatasetStatus.ANALYZING.value

        mock_db.query.return_value.filter.return_value.first.return_value = mock_dataset

        response = client.post(f"/api/v1/datasets/analyze/{fake_id}")
        assert response.status_code == 409
