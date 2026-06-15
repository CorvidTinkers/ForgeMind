"""
ForgeMind Celery Application.

Configures the Celery distributed task queue using Redis as the broker.
Workers process dataset analysis tasks asynchronously, outside the API
request lifecycle.
"""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "forgemind",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task settings
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # Result expiration (24 hours)
    result_expires=86400,

    # Task routing (future: separate queues for different agents)
    task_default_queue="forgemind.default",
    task_routes={
        "app.workers.tasks.analyze_dataset_task": {"queue": "forgemind.analysis"},
        "app.workers.tasks.run_augmentation_task": {"queue": "forgemind.augmentation"},
    },
)
