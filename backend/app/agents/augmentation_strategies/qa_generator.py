"""
ForgeMind QA Generator Strategy.

Generates new question-answer pairs from existing samples.
Produces new questions, alternate phrasings, and follow-up questions
to expand QA-style datasets.
"""

import json
import logging

from app.agents.augmentation_strategies.base import (
    AugmentationResult,
    AugmentationStrategy,
)
from app.infrastructure.generation.base import GenerationClient, GenerationRequest

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = (
    "You are a dataset augmentation assistant specializing in question-answer pair generation. "
    "Given a context or existing Q&A pair, generate a NEW question-answer pair that is "
    "related but different from the original. "
    "Return your response as valid JSON with exactly two keys: "
    '"question" and "answer". Nothing else.'
)


class QAGeneratorStrategy(AugmentationStrategy):
    """Generates new Q&A pairs from existing dataset samples."""

    @property
    def name(self) -> str:
        return "synthetic_qa"

    def execute(
        self,
        samples: list[dict],
        target_count: int,
        client: GenerationClient,
        metadata: dict | None = None,
    ) -> AugmentationResult:
        """
        Generate new QA pairs from existing samples.

        For each generation, picks a source sample and asks the LLM to
        produce a new, related QA pair.
        """
        generated = []
        failed = 0

        if not samples:
            logger.warning("No samples provided for QA generation.")
            return AugmentationResult(
                strategy_name=self.name, generated_count=0, failed_count=0
            )

        domain = self._get_domain(metadata)

        for i in range(target_count):
            source = samples[i % len(samples)]
            prompt = self._build_prompt(source, domain)

            request = GenerationRequest(
                prompt=prompt,
                system_prompt=QA_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=512,
            )

            response = client.generate(request)

            if response.text and response.finish_reason != "error":
                parsed = self._parse_qa(response.text)
                if parsed:
                    parsed["source"] = "synthetic_qa"
                    parsed["original_index"] = i % len(samples)
                    generated.append(parsed)
                else:
                    failed += 1
            else:
                failed += 1

        logger.info(
            "QA generation complete: generated=%d, failed=%d", len(generated), failed
        )

        return AugmentationResult(
            samples=generated,
            strategy_name=self.name,
            generated_count=len(generated),
            failed_count=failed,
            audit_trail={
                "prompt_template": "qa_expansion_v1",
                "temperature": 0.7,
                "generated_count": len(generated),
            },
        )

    def _build_prompt(self, sample: dict, domain: str) -> str:
        """Build a generation prompt from a source sample."""
        # Try to find Q&A structure
        question = sample.get("question", sample.get("instruction", ""))
        answer = sample.get("answer", sample.get("output", ""))

        if question and answer:
            return (
                f"Domain: {domain}\n\n"
                f"Existing Q&A pair:\n"
                f"Q: {question}\n"
                f"A: {answer}\n\n"
                f"Generate a NEW, different question-answer pair on the same topic."
            )

        # Fallback: use raw text
        text = " ".join(
            str(v) for v in sample.values() if isinstance(v, str) and v.strip()
        )
        return (
            f"Domain: {domain}\n\n"
            f"Context: {text[:500]}\n\n"
            f"Generate a question-answer pair based on this context."
        )

    def _parse_qa(self, text: str) -> dict | None:
        """Attempt to parse a JSON QA pair from generated text."""
        try:
            # Try direct JSON parse
            data = json.loads(text)
            if "question" in data and "answer" in data:
                return {"question": str(data["question"]), "answer": str(data["answer"])}
        except json.JSONDecodeError:
            pass

        # Try to find JSON block within the text
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            if "question" in data and "answer" in data:
                return {"question": str(data["question"]), "answer": str(data["answer"])}
        except (ValueError, json.JSONDecodeError):
            pass

        logger.debug("Failed to parse QA from: %s", text[:100])
        return None

    def _get_domain(self, metadata: dict | None) -> str:
        """Extract the primary domain from metadata."""
        if not metadata:
            return "general"
        domains = metadata.get("top_domains", [])
        if domains and isinstance(domains[0], dict):
            return domains[0].get("domain", "general")
        return "general"
