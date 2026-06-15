"""
ForgeMind Ollama Generation Client.

Default local LLM backend for ForgeMind. Communicates with a local
Ollama instance via its REST API. Supports Qwen, Gemma, and Llama models.
"""

import json
import logging

import requests as http_requests

from app.core.config import get_settings
from app.infrastructure.generation.base import (
    BatchGenerationResponse,
    GenerationClient,
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)


class OllamaClient(GenerationClient):
    """
    Ollama-based generation client.

    Connects to a local Ollama server via HTTP REST API.
    Default model can be configured via settings.
    """

    def __init__(self, base_url: str | None = None, default_model: str | None = None):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.default_model = default_model or settings.ollama_default_model
        self.generate_endpoint = f"{self.base_url}/api/generate"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate a single completion via the Ollama /api/generate endpoint.

        Args:
            request: The generation parameters.

        Returns:
            GenerationResponse with the generated text.
        """
        model = request.model or self.default_model

        payload = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            resp = http_requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            return GenerationResponse(
                text=data.get("response", "").strip(),
                model=data.get("model", model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                finish_reason="stop" if data.get("done", True) else "length",
            )

        except http_requests.exceptions.ConnectionError:
            logger.error("Ollama server not reachable at %s", self.base_url)
            return GenerationResponse(
                text="",
                model=model,
                finish_reason="error",
            )
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return GenerationResponse(
                text="",
                model=model,
                finish_reason="error",
            )

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> BatchGenerationResponse:
        """
        Generate completions for multiple prompts sequentially.

        Ollama does not natively support batching, so we iterate.

        Args:
            requests: List of generation requests.

        Returns:
            BatchGenerationResponse with all results.
        """
        responses = []
        total_prompt = 0
        total_completion = 0

        for req in requests:
            resp = self.generate(req)
            responses.append(resp)
            total_prompt += resp.prompt_tokens
            total_completion += resp.completion_tokens

        return BatchGenerationResponse(
            responses=responses,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
        )

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = http_requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


def get_generation_client() -> GenerationClient:
    """Factory function for dependency injection. Returns the default client."""
    return OllamaClient()
