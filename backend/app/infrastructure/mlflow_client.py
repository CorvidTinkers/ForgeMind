"""
ForgeMind MLflow Tracking Service.

Provides a service interface for experiment tracking.
In Phase 1, this is a **mocked** implementation since there
are no training runs or model experiments yet.

The interface is established now so that future phases can
simply swap in the real implementation without changing callers.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MLflowTrackingService:
    """
    MLflow tracking service interface.

    Currently mocked — logs to Python logger instead of MLflow.
    When training runs exist (Phase 3+), this will be wired
    to the real MLflow tracking server.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri
        self._enabled = False  # Mocked in Phase 1
        logger.info(
            "MLflowTrackingService initialized (mocked=%s, uri=%s)",
            not self._enabled,
            tracking_uri,
        )

    def log_dataset_analysis(
        self,
        dataset_id: str,
        metrics: dict[str, float],
        params: dict[str, str],
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Log a dataset analysis run.

        Args:
            dataset_id: The dataset identifier.
            metrics: Numeric metrics (quality_score, confidence, etc.).
            params: String parameters (domain, task, language).
            tags: Optional tags for the run.
        """
        if self._enabled:
            # Future: Real MLflow implementation
            # import mlflow
            # mlflow.set_tracking_uri(self.tracking_uri)
            # with mlflow.start_run():
            #     mlflow.log_metrics(metrics)
            #     mlflow.log_params(params)
            #     if tags:
            #         mlflow.set_tags(tags)
            pass
        else:
            logger.info(
                "[MLflow Mock] Dataset analysis logged — "
                "dataset_id=%s, metrics=%s, params=%s",
                dataset_id,
                metrics,
                params,
            )

    def log_metric(self, key: str, value: float) -> None:
        """Log a single metric."""
        if self._enabled:
            pass  # Future: mlflow.log_metric(key, value)
        else:
            logger.info("[MLflow Mock] Metric: %s = %s", key, value)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        if self._enabled:
            pass  # Future: mlflow.log_param(key, value)
        else:
            logger.info("[MLflow Mock] Param: %s = %s", key, value)


def get_mlflow_service() -> MLflowTrackingService:
    """Factory function for dependency injection."""
    from app.core.config import get_settings

    settings = get_settings()
    return MLflowTrackingService(tracking_uri=settings.mlflow_tracking_uri)
