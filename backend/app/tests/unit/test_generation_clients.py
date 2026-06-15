"""
ForgeMind Unit Tests — Generation Clients.

Tests the GenerationClient interface, OllamaClient, and stubs.
"""

import pytest

from app.infrastructure.generation.base import (
    GenerationClient,
    GenerationRequest,
    GenerationResponse,
    BatchGenerationResponse,
)
from app.infrastructure.generation.anthropic_stub import AnthropicClient
from app.infrastructure.generation.openai_stub import OpenAIClient


# ============================================
# Interface Compliance Tests
# ============================================

class TestGenerationClientInterface:
    """Verify that stub clients implement the interface."""

    def test_openai_stub_raises(self):
        client = OpenAIClient()
        request = GenerationRequest(prompt="Test")
        with pytest.raises(NotImplementedError):
            client.generate(request)

    def test_openai_stub_batch_raises(self):
        client = OpenAIClient()
        with pytest.raises(NotImplementedError):
            client.generate_batch([])

    def test_openai_stub_unavailable(self):
        client = OpenAIClient()
        assert client.is_available() is False

    def test_anthropic_stub_raises(self):
        client = AnthropicClient()
        request = GenerationRequest(prompt="Test")
        with pytest.raises(NotImplementedError):
            client.generate(request)

    def test_anthropic_stub_batch_raises(self):
        client = AnthropicClient()
        with pytest.raises(NotImplementedError):
            client.generate_batch([])

    def test_anthropic_stub_unavailable(self):
        client = AnthropicClient()
        assert client.is_available() is False


# ============================================
# GenerationRequest / Response Tests
# ============================================

class TestGenerationDataClasses:
    def test_request_defaults(self):
        req = GenerationRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.model is None

    def test_response_defaults(self):
        resp = GenerationResponse(text="World")
        assert resp.text == "World"
        assert resp.finish_reason == "stop"
        assert resp.prompt_tokens == 0

    def test_batch_response(self):
        batch = BatchGenerationResponse()
        assert batch.responses == []
        assert batch.total_prompt_tokens == 0
