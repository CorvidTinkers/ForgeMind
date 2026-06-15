"""
ForgeMind Synthetic Sample Strategy.

Generates entirely new samples using domain metadata, knowledge
extraction outputs, and dataset structure from Phase 1.
"""

import json
import logging

from app.agents.augmentation_strategies.base import (
    AugmentationResult,
    AugmentationStrategy,
)
from app.infrastructure.generation.base import GenerationClient, GenerationRequest

logger = logging.getLogger(__name__)

SYNTHETIC_SYSTEM_PROMPT = (
    "You are a dataset generation assistant. Given a domain, topic keywords, "
    "and example structure, generate a completely NEW data sample that "
    "matches the style and format of the examples. "
    "Return ONLY valid JSON matching the provided structure."
)


class SyntheticSampleStrategy(AugmentationStrategy):
    """Generates entirely new samples matching the dataset's domain and structure."""

    @property
    def name(self) -> str:
        return "synthetic_sample"

    def execute(
        self,
        samples: list[dict],
        target_count: int,
        client: GenerationClient,
        metadata: dict | None = None,
    ) -> AugmentationResult:
        """
        Generate synthetic samples using domain context and structural templates.
        """
        generated = []
        failed = 0

        if not samples:
            logger.warning("No samples provided for synthetic generation.")
            return AugmentationResult(
                strategy_name=self.name, generated_count=0, failed_count=0
            )

        domain = self._get_domain(metadata)
        keywords = self._get_keywords(metadata)
        structure = self._infer_structure(samples)

        for i in range(target_count):
            source = samples[i % len(samples)]
            prompt = self._build_prompt(source, domain, keywords, structure)

            request = GenerationRequest(
                prompt=prompt,
                system_prompt=SYNTHETIC_SYSTEM_PROMPT,
                temperature=0.9,
                max_tokens=512,
            )

            response = client.generate(request)

            if response.text and response.finish_reason != "error":
                parsed = self._parse_sample(response.text, structure)
                if parsed:
                    parsed["source"] = "synthetic_sample"
                    generated.append(parsed)
                else:
                    failed += 1
            else:
                failed += 1

        logger.info(
            "Synthetic generation complete: generated=%d, failed=%d",
            len(generated),
            failed,
        )

        return AugmentationResult(
            samples=generated,
            strategy_name=self.name,
            generated_count=len(generated),
            failed_count=failed,
            audit_trail={
                "prompt_template": "synthetic_sample_v1",
                "temperature": 0.9,
                "generated_count": len(generated),
            },
        )

    def _build_prompt(
        self, source: dict, domain: str, keywords: list[str], structure: list[str]
    ) -> str:
        """Build a synthetic generation prompt."""
        example_json = json.dumps(
            {k: v for k, v in source.items() if isinstance(v, (str, int, float))},
            indent=2,
        )

        keywords_str = ", ".join(keywords[:10]) if keywords else "general"

        return (
            f"Domain: {domain}\n"
            f"Keywords: {keywords_str}\n"
            f"Required fields: {structure}\n\n"
            f"Example sample:\n{example_json}\n\n"
            f"Generate a completely NEW sample in the same format. "
            f"Use different content but keep the same structure."
        )

    def _infer_structure(self, samples: list[dict]) -> list[str]:
        """Infer the field structure from existing samples."""
        if not samples:
            return []
        # Use keys from the first sample as the canonical structure
        return [k for k in samples[0].keys() if k != "source"]

    def _parse_sample(self, text: str, expected_keys: list[str]) -> dict | None:
        """Parse generated text into a sample dict."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            if isinstance(data, dict):
                return data
        except (ValueError, json.JSONDecodeError):
            pass

        logger.debug("Failed to parse synthetic sample: %s", text[:100])
        return None

    def _get_domain(self, metadata: dict | None) -> str:
        if not metadata:
            return "general"
        domains = metadata.get("top_domains", [])
        if domains and isinstance(domains[0], dict):
            return domains[0].get("domain", "general")
        return "general"

    def _get_keywords(self, metadata: dict | None) -> list[str]:
        if not metadata:
            return []
        knowledge = metadata.get("knowledge", {})
        keywords = knowledge.get("keywords", [])
        entities = knowledge.get("entities", [])
        return keywords + entities
