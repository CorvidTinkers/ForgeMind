"""
ForgeMind Celery Tasks.

Asynchronous task definitions for the dataset intelligence pipeline.

Flow:
    Upload API → Publish Event → Celery Worker → Full Analysis → Store Results
"""

import logging
import uuid

from app.agents.dataset_analyzer import DatasetAnalyzerAgent
from app.agents.dataset_validator import DatasetValidationAgent
from app.application.metadata_builder import MetadataBuilder
from app.domain.enums import DatasetStatus
from app.domain.models import Dataset
from app.events.redis_publisher import EventPublisher
from app.infrastructure.database import SessionLocal
from app.infrastructure.embedding_store import get_embedding_store
from app.infrastructure.mlflow_client import get_mlflow_service
from app.services.dataset_fingerprint import DatasetFingerprintService
from app.services.dataset_registry import DatasetRegistryService
from app.storage.minio_client import get_minio_client
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.analyze_dataset_task",
    max_retries=3,
    default_retry_delay=60,
)
def analyze_dataset_task(self, dataset_id: str) -> dict:
    """
    Full dataset intelligence pipeline task.

    Executed asynchronously by a Celery worker.

    Steps:
        1. Fetch dataset record from DB
        2. Download file from MinIO
        3. Validate dataset
        4. Compute fingerprint
        5. Register version
        6. Run Dataset Analyzer Agent
        7. Build and persist metadata
        8. Publish DatasetAnalyzed event
    """
    db = SessionLocal()
    event_publisher = EventPublisher()

    try:
        dataset_uuid = uuid.UUID(dataset_id)

        # 1. Fetch dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        if dataset is None:
            logger.error("Dataset not found: %s", dataset_id)
            return {"status": "error", "message": "Dataset not found"}

        # Update status
        dataset.status = DatasetStatus.VALIDATING.value
        db.commit()

        # 2. Download from MinIO
        minio_client = get_minio_client()
        # Extract object name from storage path (remove bucket prefix)
        object_name = dataset.storage_path.split("/", 1)[-1]
        raw_data = minio_client.download_file(object_name)

        # 3. Validate
        validator = DatasetValidationAgent()
        validation_result = validator.validate(raw_data, dataset.filename)

        if not validation_result.is_valid:
            dataset.status = DatasetStatus.VALIDATION_FAILED.value
            dataset.validation_errors = validation_result.errors
            db.commit()

            event_publisher.publish("DatasetValidationFailed", {
                "dataset_id": dataset_id,
                "errors": validation_result.errors,
            })

            logger.warning("Validation failed for dataset %s", dataset_id)
            return {
                "status": "validation_failed",
                "errors": validation_result.errors,
            }

        dataset.status = DatasetStatus.VALIDATED.value
        dataset.validation_errors = validation_result.warnings or []
        db.commit()

        # 4. Fingerprint
        dataset.status = DatasetStatus.FINGERPRINTING.value
        db.commit()

        import io
        fingerprint_service = DatasetFingerprintService(db)
        fingerprint_info = fingerprint_service.fingerprint_and_check(
            io.BytesIO(raw_data)
        )

        dataset.fingerprint = fingerprint_info["fingerprint"]
        dataset.status = DatasetStatus.FINGERPRINTED.value
        db.commit()

        # 5. Register version
        registry = DatasetRegistryService(db)
        registry.register_version(
            dataset_id=dataset_uuid,
            version="1.0",
            changelog="Initial upload and analysis",
        )

        # 6. Analyze
        dataset.status = DatasetStatus.ANALYZING.value
        db.commit()

        analyzer = DatasetAnalyzerAgent()
        parsed = validation_result.parsed_dataset
        metadata, dataset_embedding = analyzer.analyze(
            raw_data=raw_data,
            filename=dataset.filename,
            parsed=parsed,
        )

        # 7. Build and persist metadata
        embedding_store = get_embedding_store()
        mlflow_service = get_mlflow_service()
        builder = MetadataBuilder(db, embedding_store, mlflow_service)
        builder.build_and_persist(
            dataset_id=dataset_uuid,
            metadata=metadata,
            fingerprint_info=fingerprint_info,
            dataset_embedding=dataset_embedding,
        )

        # 8. Publish event
        event_publisher.publish("DatasetAnalyzed", {
            "dataset_id": dataset_id,
            "domain": (
                metadata.get("top_domains", [{}])[0].get("domain", "unknown")
                if metadata.get("top_domains")
                else "unknown"
            ),
            "task": metadata.get("task_detection", {}).get("task", "unknown"),
            "quality_score": metadata.get("profile", {}).get("quality_score", 0.0),
        })

        logger.info("Analysis complete for dataset %s", dataset_id)
        return {"status": "analyzed", "dataset_id": dataset_id}

    except Exception as e:
        logger.exception("Analysis failed for dataset %s: %s", dataset_id, e)

        # Update status to failed
        try:
            dataset = db.query(Dataset).filter(Dataset.id == uuid.UUID(dataset_id)).first()
            if dataset:
                dataset.status = DatasetStatus.ANALYSIS_FAILED.value
                db.commit()
        except Exception:
            pass

        event_publisher.publish("DatasetAnalysisFailed", {
            "dataset_id": dataset_id,
            "error": str(e),
        })

        # Retry with exponential backoff
        raise self.retry(exc=e)

    finally:
        db.close()


# ============================================
# Phase 2: Data Augmentation Task
# ============================================

@celery_app.task(
    bind=True,
    name="app.workers.tasks.run_augmentation_task",
    max_retries=2,
    default_retry_delay=120,
)
def run_augmentation_task(self, dataset_id: str, run_id: str) -> dict:
    """
    Full data augmentation pipeline task.

    Steps:
        1. Fetch dataset and its metadata
        2. Parse original samples
        3. Run DataAugmentationAgent (Plan → Generate → Validate → Dedup)
        4. Build training dataset (Alpaca format)
        5. Persist results to database
    """
    from datetime import datetime, timezone

    from app.agents.data_augmentation import DataAugmentationAgent
    from app.domain.enums import AugmentationStatus
    from app.domain.models import AugmentationRun, AugmentedDataset, TrainingDataset
    from app.infrastructure.generation.ollama_client import get_generation_client
    from app.services.file_parser import FileParserService
    from app.services.training_builder import TrainingDatasetBuilder

    db = SessionLocal()

    try:
        dataset_uuid = uuid.UUID(dataset_id)
        run_uuid = uuid.UUID(run_id)

        # 1. Fetch dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
        if dataset is None:
            logger.error("Dataset not found: %s", dataset_id)
            return {"status": "error", "message": "Dataset not found"}

        # Fetch augmentation run
        aug_run = db.query(AugmentationRun).filter(AugmentationRun.id == run_uuid).first()
        if aug_run is None:
            logger.error("Augmentation run not found: %s", run_id)
            return {"status": "error", "message": "Augmentation run not found"}

        # Update status
        aug_run.status = AugmentationStatus.RUNNING.value
        db.commit()

        # 2. Download and parse original samples
        minio_client = get_minio_client()
        object_name = dataset.storage_path.split("/", 1)[-1]
        raw_data = minio_client.download_file(object_name)

        parser = FileParserService()
        parsed = parser.parse(raw_data, dataset.filename)

        # Use records if available, otherwise build from sample_texts
        original_samples = parsed.records if parsed.records else [
            {"text": t} for t in parsed.sample_texts
        ]

        if not original_samples:
            aug_run.status = AugmentationStatus.FAILED.value
            aug_run.error_message = "No samples could be extracted from dataset"
            db.commit()
            return {"status": "failed", "message": "No samples extracted"}

        # Get metadata from Phase 1
        metadata = dataset.metadata_ or {}

        # 3. Run augmentation agent
        client = get_generation_client()
        agent = DataAugmentationAgent(generation_client=client)
        augmented_samples, report = agent.augment(
            samples=original_samples,
            metadata=metadata,
        )

        # 4. Build training dataset
        aug_run.status = AugmentationStatus.BUILDING.value
        db.commit()

        builder = TrainingDatasetBuilder()
        training_records = builder.build(augmented_samples, format_name="alpaca")

        # 5. Persist results
        # Save augmented dataset
        aug_dataset = AugmentedDataset(
            augmentation_run_id=run_uuid,
            samples=augmented_samples,
            sample_count=len(augmented_samples),
        )
        db.add(aug_dataset)

        # Save training dataset
        training_ds = TrainingDataset(
            augmentation_run_id=run_uuid,
            dataset_id=dataset_uuid,
            format_name="alpaca",
            records=training_records,
            record_count=len(training_records),
        )
        db.add(training_ds)

        # Update augmentation run
        aug_run.status = AugmentationStatus.COMPLETED.value
        aug_run.completed_at = datetime.now(timezone.utc)
        aug_run.report = {
            "original_samples": report.original_samples,
            "generated_samples": report.generated_samples,
            "accepted_samples": report.accepted_samples,
            "rejected_samples": report.rejected_samples,
            "duplicates_removed": report.duplicates_removed,
            "final_samples": report.final_samples,
            "quality_score": report.quality_score,
            "strategies_used": report.strategies_used,
            "rejection_reasons": report.rejection_reasons,
            "rejected_examples": report.rejected_examples,
            "sanity_report": report.sanity_report,
            "quality_report": report.quality_report,
            "deduplication_report": report.deduplication_report,
        }
        aug_run.audit_trail = report.audit_trails

        db.commit()

        # Publish event
        event_publisher = EventPublisher()
        event_publisher.publish("AugmentationCompleted", {
            "dataset_id": dataset_id,
            "run_id": run_id,
            "final_samples": report.final_samples,
            "quality_score": report.quality_score,
        })

        logger.info(
            "Augmentation complete: dataset=%s, run=%s, final_samples=%d",
            dataset_id,
            run_id,
            report.final_samples,
        )

        return {
            "status": "completed",
            "dataset_id": dataset_id,
            "run_id": run_id,
            "final_samples": report.final_samples,
        }

    except Exception as e:
        logger.exception("Augmentation failed: dataset=%s, run=%s: %s", dataset_id, run_id, e)

        try:
            aug_run = db.query(AugmentationRun).filter(
                AugmentationRun.id == uuid.UUID(run_id)
            ).first()
            if aug_run:
                aug_run.status = AugmentationStatus.FAILED.value
                aug_run.error_message = str(e)
                db.commit()
        except Exception:
            pass

        raise self.retry(exc=e)

    finally:
        db.close()

