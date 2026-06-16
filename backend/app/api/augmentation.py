"""
ForgeMind Augmentation API Endpoints.

    POST   /datasets/{id}/augment        Trigger augmentation
    GET    /augmentation/{run_id}         Get augmentation run status
    GET    /augmentation/{run_id}/report  Get detailed augmentation report
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.domain.enums import AugmentationStatus, DatasetStatus
from app.domain.models import AugmentationRun, Dataset
from app.domain.schemas import (
    AugmentationReportResponse,
    AugmentationRunResponse,
    AugmentResponse,
)
from app.workers.tasks import run_augmentation_task

logger = logging.getLogger(__name__)

router = APIRouter(tags=["augmentation"])


@router.post(
    "/datasets/{dataset_id}/augment",
    response_model=AugmentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger dataset augmentation",
    description="Triggers an asynchronous augmentation run for an analyzed dataset.",
)
async def augment_dataset(
    dataset_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AugmentResponse:
    """
    Trigger the data augmentation pipeline.

    The dataset must be in 'analyzed' status (Phase 1 complete).
    Creates an AugmentationRun record and submits a Celery task.
    """
    # Check dataset exists and is analyzed
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found.",
        )

    if dataset.status != DatasetStatus.ANALYZED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset must be analyzed before augmentation (current status: {dataset.status}).",
        )

    # Check for existing active augmentation run
    active_run = (
        db.query(AugmentationRun)
        .filter(
            AugmentationRun.dataset_id == dataset_id,
            AugmentationRun.status.in_([
                AugmentationStatus.PENDING.value,
                AugmentationStatus.RUNNING.value,
                AugmentationStatus.PLANNING.value,
            ]),
        )
        .first()
    )
    if active_run:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An augmentation run is already in progress (run_id: {active_run.id}).",
        )

    # Create augmentation run
    aug_run = AugmentationRun(
        dataset_id=dataset_id,
        status=AugmentationStatus.PENDING.value,
    )
    db.add(aug_run)
    db.commit()
    db.refresh(aug_run)

    # Submit Celery task
    run_augmentation_task.delay(str(dataset_id), str(aug_run.id))

    logger.info(
        "Augmentation triggered: dataset=%s, run=%s", dataset_id, aug_run.id
    )

    return AugmentResponse(
        run_id=aug_run.id,
        dataset_id=dataset_id,
        status=AugmentationStatus.PENDING.value,
        message="Augmentation task submitted. Check status via GET /augmentation/{run_id}.",
    )


@router.get(
    "/augmentation/{run_id}",
    response_model=AugmentationRunResponse,
    summary="Get augmentation run status",
    description="Returns the status and metadata of an augmentation run.",
)
async def get_augmentation_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AugmentationRunResponse:
    """Get the status of an augmentation run."""
    aug_run = db.query(AugmentationRun).filter(AugmentationRun.id == run_id).first()
    if aug_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Augmentation run {run_id} not found.",
        )

    return AugmentationRunResponse(
        id=aug_run.id,
        dataset_id=aug_run.dataset_id,
        status=aug_run.status,
        config=aug_run.config,
        report=aug_run.report,
        created_at=aug_run.created_at,
        completed_at=aug_run.completed_at,
    )


@router.get(
    "/augmentation/{run_id}/report",
    response_model=AugmentationReportResponse,
    summary="Get augmentation report",
    description="Returns the detailed augmentation report with quality metrics.",
)
async def get_augmentation_report(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AugmentationReportResponse:
    """Get the detailed report of a completed augmentation run."""
    aug_run = db.query(AugmentationRun).filter(AugmentationRun.id == run_id).first()
    if aug_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Augmentation run {run_id} not found.",
        )

    if aug_run.status != AugmentationStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Report not available. Augmentation status: {aug_run.status}.",
        )

    report_data = aug_run.report or {}

    return AugmentationReportResponse(
        run_id=aug_run.id,
        original_samples=report_data.get("original_samples", 0),
        generated_samples=report_data.get("generated_samples", 0),
        accepted_samples=report_data.get("accepted_samples", 0),
        rejected_samples=report_data.get("rejected_samples", 0),
        duplicates_removed=report_data.get("duplicates_removed", 0),
        final_samples=report_data.get("final_samples", 0),
        quality_score=report_data.get("quality_score", 0.0),
        strategies_used=report_data.get("strategies_used", []),
        rejection_reasons=report_data.get("rejection_reasons", {}),
        rejected_examples=report_data.get("rejected_examples", []),
        sanity_report=report_data.get("sanity_report", {}),
        quality_report=report_data.get("quality_report", {}),
        deduplication_report=report_data.get("deduplication_report", {}),
        audit_trails=aug_run.audit_trail or [],
    )
