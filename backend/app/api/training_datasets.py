"""
ForgeMind Training Dataset API Endpoints.

    GET    /training-dataset/{id}                    Get training dataset metadata
    GET    /training-dataset/{id}/download            Download training dataset
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.domain.models import TrainingDataset
from app.domain.schemas import TrainingDatasetResponse
from app.services.export_service import ExportService
from app.services.training_builder import TrainingDatasetBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-dataset", tags=["training-datasets"])


@router.get(
    "/{dataset_id}",
    response_model=TrainingDatasetResponse,
    summary="Get training dataset metadata",
    description="Returns metadata for a compiled training dataset.",
)
async def get_training_dataset(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> TrainingDatasetResponse:
    """Get metadata for a training dataset."""
    training_ds = (
        db.query(TrainingDataset)
        .filter(TrainingDataset.id == dataset_id)
        .first()
    )
    if training_ds is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training dataset {dataset_id} not found.",
        )

    return TrainingDatasetResponse(
        id=training_ds.id,
        dataset_id=training_ds.dataset_id,
        augmentation_run_id=training_ds.augmentation_run_id,
        format_name=training_ds.format_name,
        record_count=training_ds.record_count,
        created_at=training_ds.created_at,
    )


@router.get(
    "/{dataset_id}/download",
    summary="Download training dataset",
    description="Download the training dataset in JSON or JSONL format.",
)
async def download_training_dataset(
    dataset_id: uuid.UUID,
    format: str = Query(
        default="alpaca",
        description="Training format: 'alpaca' or 'chatml'",
    ),
    file_format: str = Query(
        default="json",
        description="File format: 'json' or 'jsonl'",
    ),
    db: Session = Depends(get_db),
):
    """
    Download a training dataset in the specified format.

    If the requested format differs from the stored format,
    re-converts on the fly using the TrainingDatasetBuilder.
    """
    # Find training dataset
    training_ds = (
        db.query(TrainingDataset)
        .filter(TrainingDataset.id == dataset_id)
        .first()
    )
    if training_ds is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training dataset {dataset_id} not found.",
        )

    records = training_ds.records or []

    # Re-convert if different format requested
    if format != training_ds.format_name and records:
        builder = TrainingDatasetBuilder()
        try:
            # Get the raw augmented samples from the stored records
            # and re-convert to the requested format
            records = builder.build(records, format_name=format)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    # Export
    export_service = ExportService()
    try:
        data, content_type = export_service.export(records, file_format=file_format)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    filename = f"training_dataset_{dataset_id}_{format}.{file_format}"

    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
