"""
ForgeMind Validation Pipeline Unit Tests.

Specifically targets the problematic instruction patterns
identified in Phase 2.1 testing:
  - "Who are computers?"
  - "Who are Quantum computers?"
  - "Who are machine learning algorithms?"
  - "What is quantum computing?"

Tests the InstructionSanityValidator, ValidationPipeline, and
the full integration of filtering into AugmentationReport metrics.
"""

import pytest

from app.agents.augmentation_strategies.sanity_validator import (
    InstructionSanityValidator,
    SanityReport,
)
from app.agents.augmentation_strategies.validation_pipeline import (
    ValidationPipeline,
    ValidationPipelineReport,
)


# ============================================================
# InstructionSanityValidator Unit Tests
# ============================================================


class TestInstructionSanityValidator:
    """Unit tests for the InstructionSanityValidator."""

    @pytest.fixture
    def validator(self):
        return InstructionSanityValidator()

    # ----- Who-targets-inanimate rejections -----

    @pytest.mark.parametrize(
        "instruction",
        [
            "Who are computers?",
            "Who are Quantum computers?",
            "Who are machine learning algorithms?",
            "Who is quantum computing?",
            "Who are natural language processing systems?",
            "Who is deep learning?",
            "Who are neural networks?",
            "Who is the internet?",
        ],
    )
    def test_rejects_who_targets_inanimate(self, validator, instruction):
        """'Who' questions targeting inanimate concepts must be rejected."""
        samples = [{"instruction": instruction}]
        valid, rejected, report = validator.validate(samples)

        assert len(valid) == 0, f"Should reject: {instruction}"
        assert len(rejected) == 1
        assert report.failed == 1
        assert report.passed == 0
        assert "who_targets_inanimate" in report.rejection_reasons

    @pytest.mark.parametrize(
        "instruction",
        [
            "Who are computers?",
            "Who are Quantum computers?",
            "Who are machine learning algorithms?",
        ],
    )
    def test_rejects_who_inanimate_via_question_key(self, validator, instruction):
        """Samples using the 'question' key should also be caught."""
        samples = [{"question": instruction, "answer": "Some answer"}]
        valid, rejected, report = validator.validate(samples)

        assert len(valid) == 0, f"Should reject (question key): {instruction}"
        assert len(rejected) == 1
        assert "who_targets_inanimate" in report.rejection_reasons

    # ----- Valid instructions that should pass -----

    @pytest.mark.parametrize(
        "instruction",
        [
            "What is quantum computing?",
            "Explain how neural networks work.",
            "How does machine learning differ from deep learning?",
            "Describe the architecture of a transformer model.",
            "List the main types of supervised learning algorithms.",
            "Who is Alan Turing?",
            "Who was Ada Lovelace?",
            "Who are the founders of Google?",
        ],
    )
    def test_accepts_valid_instructions(self, validator, instruction):
        """Well-formed instructions must be accepted."""
        samples = [{"instruction": instruction}]
        valid, rejected, report = validator.validate(samples)

        assert len(valid) == 1, f"Should accept: {instruction}"
        assert len(rejected) == 0
        assert report.passed == 1
        assert report.failed == 0

    # ----- Too-short rejections -----

    def test_rejects_too_short(self, validator):
        """Very short instructions must be rejected."""
        samples = [{"instruction": "Hi"}]
        valid, rejected, report = validator.validate(samples)

        assert len(valid) == 0
        assert "too_short" in report.rejection_reasons

    # ----- Declarative rejections -----

    def test_rejects_purely_declarative(self, validator):
        """Plain declarative statements must be rejected."""
        samples = [{"instruction": "AI has become more advanced in recent years."}]
        valid, rejected, report = validator.validate(samples)

        assert len(valid) == 0
        assert "declarative_not_instruction" in report.rejection_reasons

    # ----- Report structure -----

    def test_report_structure(self, validator):
        """Report must contain all expected fields with correct counts."""
        samples = [
            {"instruction": "What is quantum computing?"},         # valid
            {"instruction": "Who are computers?"},                 # rejected
            {"instruction": "Explain neural networks."},           # valid
            {"instruction": "Who are machine learning algorithms?"},  # rejected
            {"instruction": "Hi"},                                 # too_short
        ]
        valid, rejected, report = validator.validate(samples)

        assert report.total_checked == 5
        assert report.passed == 2
        assert report.failed == 3
        assert len(valid) == 2
        assert len(rejected) == 3
        assert report.rejection_reasons.get("who_targets_inanimate", 0) == 2
        assert report.rejection_reasons.get("too_short", 0) == 1
        assert len(report.rejected_examples) == 3

    # ----- Batch with mixed keys -----

    def test_mixed_key_samples(self, validator):
        """Validator must handle samples with different key names."""
        samples = [
            {"instruction": "Who are computers?"},
            {"question": "Who are Quantum computers?", "answer": "Some answer"},
            {"instruction": "What is deep learning?"},
            {"question": "Explain backpropagation.", "answer": "A method..."},
        ]
        valid, rejected, report = validator.validate(samples)

        assert report.passed == 2
        assert report.failed == 2
        assert "who_targets_inanimate" in report.rejection_reasons


# ============================================================
# ValidationPipeline Integration Tests
# ============================================================


class TestValidationPipeline:
    """Tests for the full ValidationPipeline (Sanity → Quality → Dedup)."""

    @pytest.fixture
    def pipeline(self):
        return ValidationPipeline()

    def test_pipeline_rejects_invalid_instructions(self, pipeline):
        """Invalid instructions must be filtered out by the pipeline."""
        generated = [
            {"instruction": "Who are computers?", "output": "Computers are..."},
            {"instruction": "Who are Quantum computers?", "output": "Quantum..."},
            {"instruction": "What is quantum computing?", "output": "QC is..."},
            {"instruction": "Who are machine learning algorithms?", "output": "ML..."},
        ]
        originals = [
            {"instruction": "What is AI?", "output": "AI is..."},
            {"instruction": "Explain deep learning.", "output": "DL is..."},
        ]

        final, report = pipeline.run(
            generated_samples=generated,
            original_samples=originals,
        )

        # The 3 "Who are <inanimate>?" samples must be gone
        final_instructions = [s.get("instruction", "") for s in final]
        assert "Who are computers?" not in final_instructions
        assert "Who are Quantum computers?" not in final_instructions
        assert "Who are machine learning algorithms?" not in final_instructions

        # Report metrics must be non-zero
        assert report.rejected_samples > 0
        assert report.accepted_samples >= 0
        assert "who_targets_inanimate" in report.rejection_reasons
        assert report.rejection_reasons["who_targets_inanimate"] >= 3

        # Sanity sub-report must reflect the rejections
        assert report.sanity_report.get("failed", 0) >= 3

    def test_pipeline_report_totals_add_up(self, pipeline):
        """accepted + rejected must equal total_input."""
        generated = [
            {"instruction": "What is AI?", "output": "AI is..."},
            {"instruction": "Who are computers?", "output": "Computers..."},
            {"instruction": "Explain ML.", "output": "ML is..."},
        ]
        originals = [
            {"instruction": "What is deep learning?", "output": "DL is..."},
        ]

        final, report = pipeline.run(
            generated_samples=generated,
            original_samples=originals,
        )

        assert report.total_input == 3
        assert report.accepted_samples + report.rejected_samples == report.total_input

    def test_pipeline_empty_input(self, pipeline):
        """Pipeline must handle empty generated samples gracefully."""
        final, report = pipeline.run(
            generated_samples=[],
            original_samples=[{"instruction": "What is AI?", "output": "..."}],
        )

        assert final == []
        assert report.total_input == 0
        assert report.accepted_samples == 0
        assert report.rejected_samples == 0
