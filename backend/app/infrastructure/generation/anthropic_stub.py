"""
ForgeMind Anthropic Generation Client — Stub.

Placeholder implementation for future Anthropic Claude API integration.
Not active in Phase 2. ForgeMind defaults to OllamaClient.
"""

import logging

from app.infrastructure.generation.base import (
    BatchGenerationResponse,
    GenerationClient,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)


class AnthropicClient(GenerationClient):
    """
    Anthropic Claude API stub client.

    This is a placeholder for future integration.
    All methods raise NotImplementedError until implemented.
    """

    def __init__(self, api_key: str = "", model: str = "claude-3-haiku"):
        self.api_key = api_key
        self.default_model = model
        logger.info("Anthropic client stub initialized (not active)")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError(
            "Anthropic client is a stub. Use OllamaClient as the default backend."
        )

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> BatchGenerationResponse:
        raise NotImplementedError(
            "Anthropic client is a stub. Use OllamaClient as the default backend."
        )

    def is_available(self) -> bool:
        return False
