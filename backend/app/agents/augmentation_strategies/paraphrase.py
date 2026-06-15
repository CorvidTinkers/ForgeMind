"""
ForgeMind Paraphrase Strategy.

Generates semantic variations of existing samples while preserving
their original meaning. Increases dataset diversity without changing
the underlying information.
"""

import logging
import random

from app.agents.augmentation_strategies.base import (
    AugmentationResult,
    AugmentationStrategy,
)
from app.infrastructure.generation.base import GenerationClient, GenerationRequest

logger = logging.getLogger(__name__)

PARAPHRASE_SYSTEM_PROMPT = (
    "You are a paraphrasing assistant. Rewrite the given text to create "
    "a semantically equivalent version with different wording and structure. "
    "Preserve all factual information and meaning. "
    "Return ONLY the paraphrased text, nothing else."
)


class ParaphraseStrategy(AugmentationStrategy):
    """Generates paraphrased variations of existing samples."""

    @property
    def name(self) -> str:
        return "paraphrase"

    def execute(
        self,
        samples: list[dict],
        target_count: int,
        client: GenerationClient,
        metadata: dict | None = None,
    ) -> AugmentationResult:
        """
        Paraphrase existing samples to generate new variations.

        Cycles through original samples, generating one paraphrase per call
        until the target count is reached.
        """
        generated = []
        failed = 0

        if not samples:
            logger.warning("No samples provided for paraphrasing.")
            return AugmentationResult(
                strategy_name=self.name, generated_count=0, failed_count=0
            )

        # Extract text content from samples
        text_samples = self._extract_texts(samples)
        if not text_samples:
            logger.warning("Could not extract text from samples.")
            return AugmentationResult(
                strategy_name=self.name, generated_count=0, failed_count=0
            )

        for i in range(target_count):
            # Cycle through original samples
            original = text_samples[i % len(text_samples)]

            prompt = f"Paraphrase the following text:\n\n{original}"

            request = GenerationRequest(
                prompt=prompt,
                system_prompt=PARAPHRASE_SYSTEM_PROMPT,
                temperature=0.8,
                max_tokens=512,
            )

            response = client.generate(request)

            if response.text and response.finish_reason != "error":
                generated.append({
                    "text": response.text,
                    "source": "paraphrase",
                    "original_index": i % len(text_samples),
                })
            else:
                failed += 1

        logger.info(
            "Paraphrase complete: generated=%d, failed=%d", len(generated), failed
        )

        return AugmentationResult(
            samples=generated,
            strategy_name=self.name,
            generated_count=len(generated),
            failed_count=failed,
            audit_trail={
                "prompt_template": "paraphrase_v1",
                "temperature": 0.8,
                "generated_count": len(generated),
            },
        )

    def _extract_texts(self, samples: list[dict]) -> list[str]:
        """Extract text content from sample dicts."""
        texts = []
        # Try common text field names
        text_keys = ["text", "content", "question", "instruction", "input"]
        for sample in samples:
            for key in text_keys:
                if key in sample and isinstance(sample[key], str) and sample[key].strip():
                    texts.append(sample[key].strip())
                    break
            else:
                # Concatenate all string values
                parts = [
                    str(v).strip()
                    for v in sample.values()
                    if isinstance(v, str) and v.strip()
                ]
                if parts:
                    texts.append(" ".join(parts))
        return texts
