"""
ForgeMind Redis Event Publisher.

Implements a lightweight event bus using Redis Pub/Sub.
Future agents subscribe to these events to react to dataset lifecycle changes.

Events:
    - DatasetUploaded
    - DatasetValidated
    - DatasetAnalyzed
    - DatasetAnalysisFailed
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EventPublisher:
    """Redis-based event publisher for the ForgeMind event bus."""

    def __init__(self) -> None:
        settings = get_settings()
        self._redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """
        Publish an event to the Redis event bus.

        Args:
            event_type: The event name (e.g., "DatasetUploaded").
            payload: Event data as a dictionary.
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        channel = f"forgemind.events.{event_type}"
        message = json.dumps(event, default=str)
        self._redis.publish(channel, message)
        logger.info("Published event: %s on channel: %s", event_type, channel)

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._redis.ping()
        except redis.ConnectionError:
            return False


def get_event_publisher() -> EventPublisher:
    """Factory function for dependency injection."""
    return EventPublisher()
