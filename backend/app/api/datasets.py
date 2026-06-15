"""
ForgeMind Dataset API Endpoints.

Exposes the dataset intelligence layer via REST API:

    POST   /datasets/upload              Upload a dataset file
    POST   /datasets/analyze/{id}        Trigger analysis workflow
    GET    /datasets/{id}                Get full intelligence report
    GET    /datasets/{id}/status         Get processing status
    GET    /datasets/                    List all datasets
"""

import io
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.domain.enums import DatasetStatus
from app.domain.models import Dataset, DatasetVersion
from app.domain.schemas import (
    AnalyzeResponse,
    DatasetMetadata,
    DatasetResponse,
    DatasetVersionInfo,
    StatusResponse,
    UploadResponse,
)
from app.events.redis_publisher import get_event_publisher
from app.storage.minio_client import get_minio_client
from app.workers.tasks import analyze_dataset_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "txt", "csv", "json"}
# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a dataset file",
    description="Upload a PDF, TXT, CSV, or JSON file. The file is stored in MinIO and a dataset record is created.",
)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Upload a dataset file to ForgeMind.

    Steps:
    1. Validate file extension and size
    2. Upload to MinIO
    3. Create dataset record in PostgreSQL
    4. Publish DatasetUploaded event
    """
    # Validate extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: .{extension}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)} MB.",
        )

    # Generate dataset ID and object path
    dataset_id = uuid.uuid4()
    object_name = f"datasets/{dataset_id}/{file.filename}"

    # Upload to MinIO
    try:
        minio_client = get_minio_client()
        storage_path = minio_client.upload_file(
            object_name=object_name,
            data=io.BytesIO(content),
            length=len(content),
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        logger.error("MinIO upload failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to store file. Please try again.",
        )

    # Create dataset record
    dataset = Dataset(
        id=dataset_id,
        filename=file.filename,
        original_filename=file.filename,
        storage_path=storage_path,
        file_size=len(content),
        mime_type=file.content_type,
        status=DatasetStatus.UPLOADED.value,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # Publish event
    try:
        publisher = get_event_publisher()
        publisher.publish("DatasetUploaded", {
            "dataset_id": str(dataset_id),
            "filename": file.filename,
            "file_size": len(content),
        })
    except Exception as e:
        logger.warning("Failed to publish DatasetUploaded event: %s", e)

    logger.info("Dataset uploaded: id=%s, filename=%s", dataset_id, file.filename)
    return UploadResponse(dataset_id=dataset_id, status="uploaded")


@router.post(
    "/analyze/{dataset_id}",
    response_model=AnalyzeResponse,
    summary="Trigger dataset analysis",
    description="Triggers the asynchronous analysis workflow for an uploaded dataset.",
)
async def analyze_dataset(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    """
    Trigger the dataset analysis pipeline.

    Submits a Celery task that runs:
    Validate → Fingerprint → Register → Analyze → Store Metadata
    """
    # Check dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found.",
        )

    # Check if already being analyzed
    active_statuses = {
        DatasetStatus.VALIDATING.value,
        DatasetStatus.FINGERPRINTING.value,
        DatasetStatus.ANALYZING.value,
    }
    if dataset.status in active_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset is currently being processed (status: {dataset.status}).",
        )

    # Submit Celery task
    analyze_dataset_task.delay(str(dataset_id))

    logger.info("Analysis triggered for dataset: %s", dataset_id)
    return AnalyzeResponse(
        dataset_id=dataset_id,
        status=dataset.status,
        message="Analysis task submitted. Check status via GET /datasets/{id}/status.",
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset intelligence report",
    description="Returns the full dataset intelligence report including metadata, profile, and knowledge extraction.",
)
async def get_dataset(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    """Retrieve the full dataset intelligence report."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found.",
        )

    # Fetch versions
    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.created_at.asc())
        .all()
    )

    version_infos = [
        DatasetVersionInfo(
            version=v.version,
            parent_version_id=v.parent_version_id,
            changelog=v.changelog,
            created_at=v.created_at,
        )
        for v in versions
    ]

    # Parse metadata from JSONB
    metadata = None
    if dataset.metadata_:
        try:
            metadata = DatasetMetadata(**dataset.metadata_)
        except Exception:
            metadata = DatasetMetadata()

    return DatasetResponse(
        id=dataset.id,
        filename=dataset.filename,
        original_filename=dataset.original_filename,
        storage_path=dataset.storage_path,
        fingerprint=dataset.fingerprint,
        file_size=dataset.file_size,
        mime_type=dataset.mime_type,
        status=dataset.status,
        metadata=metadata,
        validation_errors=dataset.validation_errors,
        versions=version_infos,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


@router.get(
    "/{dataset_id}/status",
    response_model=StatusResponse,
    summary="Get dataset processing status",
    description="Returns the current processing status of a dataset.",
)
async def get_dataset_status(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> StatusResponse:
    """Get the current processing status of a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found.",
        )

    return StatusResponse(
        dataset_id=dataset.id,
        status=dataset.status,
        validation_errors=dataset.validation_errors,
    )


@router.get(
    "/",
    response_model=list[DatasetResponse],
    summary="List all datasets",
    description="Returns a list of all datasets with their metadata.",
)
async def list_datasets(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
) -> list[DatasetResponse]:
    """List all datasets with pagination."""
    datasets = (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    results = []
    for dataset in datasets:
        metadata = None
        if dataset.metadata_:
            try:
                metadata = DatasetMetadata(**dataset.metadata_)
            except Exception:
                metadata = DatasetMetadata()

        results.append(
            DatasetResponse(
                id=dataset.id,
                filename=dataset.filename,
                original_filename=dataset.original_filename,
                storage_path=dataset.storage_path,
                fingerprint=dataset.fingerprint,
                file_size=dataset.file_size,
                mime_type=dataset.mime_type,
                status=dataset.status,
                metadata=metadata,
                validation_errors=dataset.validation_errors,
                versions=[],
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
            )
        )

    return results
