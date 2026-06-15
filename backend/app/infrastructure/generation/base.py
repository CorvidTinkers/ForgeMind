"""
ForgeMind Generation Client — Abstract Interface.

Defines the provider-agnostic contract for all LLM generation backends.
Generators must communicate ONLY through this interface, never directly
calling a specific provider (OpenAI, Anthropic, Ollama, etc.).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """
    A single generation request to an LLM backend.

    Attributes:
        prompt: The full prompt text to send.
        system_prompt: Optional system-level instruction.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens in the response.
        model: Model name override (uses client default if None).
    """

    prompt: str
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    model: str | None = None


@dataclass
class GenerationResponse:
    """
    Response from an LLM generation call.

    Attributes:
        text: The generated text content.
        model: The model that produced this response.
        prompt_tokens: Number of tokens in the prompt (if available).
        completion_tokens: Number of tokens in the completion (if available).
        finish_reason: Why generation stopped (e.g., 'stop', 'length').
    """

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


@dataclass
class BatchGenerationResponse:
    """Response from a batch generation call."""

    responses: list[GenerationResponse] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


class GenerationClient(ABC):
    """
    Abstract interface for LLM generation backends.

    All augmentation generators (QAGenerator, ParaphraseGenerator, etc.)
    depend on this interface, never on concrete provider implementations.
    """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate a single completion.

        Args:
            request: The generation request parameters.

        Returns:
            GenerationResponse with the generated text.
        """
        ...

    @abstractmethod
    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> BatchGenerationResponse:
        """
        Generate completions for multiple prompts.

        Default implementation calls generate() sequentially.
        Concrete clients may override for true batching.

        Args:
            requests: List of generation requests.

        Returns:
            BatchGenerationResponse with all results.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the generation backend is reachable."""
        ...
