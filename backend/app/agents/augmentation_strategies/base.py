"""
ForgeMind Augmentation Strategy — Abstract Base Class.

Defines the extensible contract for all augmentation strategies.
Each strategy implements `execute()` to produce new samples using
the GenerationClient interface.

This pattern replaces `if strategy == ...` blocks and makes
Phase 3 and Phase 4 significantly cleaner to extend.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.infrastructure.generation.base import GenerationClient

logger = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    """
    Output from a single augmentation strategy execution.

    Attributes:
        samples: The generated samples (list of dicts).
        strategy_name: Name of the strategy that produced these samples.
        generated_count: Number of samples successfully generated.
        failed_count: Number of generation attempts that failed.
        audit_trail: Generation parameters for traceability.
    """

    samples: list[dict] = field(default_factory=list)
    strategy_name: str = ""
    generated_count: int = 0
    failed_count: int = 0
    audit_trail: dict = field(default_factory=dict)


class AugmentationStrategy(ABC):
    """
    Abstract base class for all augmentation strategies.

    Concrete implementations:
        - ParaphraseStrategy
        - QAGeneratorStrategy
        - SyntheticSampleStrategy
        - ConversationStrategy (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy identifier (e.g., 'paraphrase', 'synthetic_qa')."""
        ...

    @abstractmethod
    def execute(
        self,
        samples: list[dict],
        target_count: int,
        client: GenerationClient,
        metadata: dict | None = None,
    ) -> AugmentationResult:
        """
        Execute the augmentation strategy.

        Args:
            samples: Original dataset samples to augment from.
            target_count: Number of new samples to generate.
            client: The GenerationClient to use for LLM calls.
            metadata: Optional Phase 1 metadata for context.

        Returns:
            AugmentationResult with the generated samples and audit trail.
        """
        ...
