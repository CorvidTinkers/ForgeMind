"""
ForgeMind Training Dataset Builder.

Converts augmented datasets into standard instruction-tuning formats:
- Alpaca Format: {"instruction": "...", "input": "...", "output": "..."}
- ChatML Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Designed to be extensible for future formats.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FormatConverter(ABC):
    """Abstract base class for format converters."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        ...

    @abstractmethod
    def convert(self, samples: list[dict]) -> list[dict]:
        """Convert samples to the target format."""
        ...


class AlpacaConverter(FormatConverter):
    """Converts samples to Alpaca instruction-tuning format."""

    @property
    def format_name(self) -> str:
        return "alpaca"

    def convert(self, samples: list[dict]) -> list[dict]:
        """
        Convert samples to Alpaca format.

        Maps common field names to instruction/input/output.
        """
        converted = []
        for sample in samples:
            record = {
                "instruction": self._get_instruction(sample),
                "input": self._get_input(sample),
                "output": self._get_output(sample),
            }
            # Only include records with at least instruction and output
            if record["instruction"] and record["output"]:
                converted.append(record)

        logger.info("Alpaca conversion: %d/%d samples converted", len(converted), len(samples))
        return converted

    def _get_instruction(self, sample: dict) -> str:
        for key in ["instruction", "question", "prompt", "text"]:
            if key in sample and isinstance(sample[key], str) and sample[key].strip():
                return sample[key].strip()
        return ""

    def _get_input(self, sample: dict) -> str:
        for key in ["input", "context"]:
            if key in sample and isinstance(sample[key], str) and sample[key].strip():
                return sample[key].strip()
        return ""

    def _get_output(self, sample: dict) -> str:
        for key in ["output", "answer", "response", "label"]:
            if key in sample and isinstance(sample[key], str) and sample[key].strip():
                return sample[key].strip()
        return ""


class ChatMLConverter(FormatConverter):
    """Converts samples to ChatML conversation format."""

    @property
    def format_name(self) -> str:
        return "chatml"

    def convert(self, samples: list[dict]) -> list[dict]:
        """
        Convert samples to ChatML format.

        Each sample becomes a list of role/content message dicts.
        """
        converted = []
        for sample in samples:
            messages = self._build_messages(sample)
            if len(messages) >= 2:
                converted.append({"messages": messages})

        logger.info("ChatML conversion: %d/%d samples converted", len(converted), len(samples))
        return converted

    def _build_messages(self, sample: dict) -> list[dict]:
        messages = []

        # Try to find user/assistant pattern
        user_content = ""
        assistant_content = ""

        for key in ["question", "instruction", "prompt", "text", "input"]:
            if key in sample and isinstance(sample[key], str) and sample[key].strip():
                user_content = sample[key].strip()
                break

        for key in ["answer", "output", "response", "label"]:
            if key in sample and isinstance(sample[key], str) and sample[key].strip():
                assistant_content = sample[key].strip()
                break

        if user_content:
            messages.append({"role": "user", "content": user_content})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        return messages


# Converter registry
FORMAT_CONVERTERS: dict[str, type[FormatConverter]] = {
    "alpaca": AlpacaConverter,
    "chatml": ChatMLConverter,
}


class TrainingDatasetBuilder:
    """
    Builds training-ready datasets from augmented samples.

    Supports multiple output formats and is extensible via
    the FormatConverter pattern.
    """

    def build(
        self,
        samples: list[dict],
        format_name: str = "alpaca",
    ) -> list[dict]:
        """
        Convert augmented samples to a training-ready format.

        Args:
            samples: The augmented dataset samples.
            format_name: Target format ('alpaca' or 'chatml').

        Returns:
            List of dicts in the target format.

        Raises:
            ValueError: If the format is not supported.
        """
        converter_class = FORMAT_CONVERTERS.get(format_name)
        if converter_class is None:
            supported = ", ".join(FORMAT_CONVERTERS.keys())
            raise ValueError(
                f"Unsupported format: '{format_name}'. Supported: {supported}"
            )

        converter = converter_class()
        result = converter.convert(samples)

        logger.info(
            "Training dataset built: format=%s, samples=%d",
            format_name,
            len(result),
        )

        return result
