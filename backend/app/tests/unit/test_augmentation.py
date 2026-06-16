"""
ForgeMind Unit Tests — Augmentation Components.

Tests the augmentation planner, strategies, training builder,
quality validator, deduplication engine, and export service.
"""

import json

import pytest

from app.agents.augmentation_planner import AugmentationPlanner
from app.agents.augmentation_strategies.base import AugmentationResult
from app.agents.augmentation_strategies.deduplication_engine import DeduplicationEngine
from app.agents.augmentation_strategies.paraphrase import ParaphraseStrategy
from app.agents.augmentation_strategies.qa_generator import QAGeneratorStrategy
from app.agents.augmentation_strategies.quality_validator import QualityValidator
from app.agents.augmentation_strategies.synthetic_sample import SyntheticSampleStrategy
from app.infrastructure.generation.base import (
    GenerationClient,
    GenerationRequest,
    GenerationResponse,
    BatchGenerationResponse,
)
from app.services.export_service import ExportService
from app.services.training_builder import TrainingDatasetBuilder


# ============================================
# Mock GenerationClient for testing
# ============================================

class MockGenerationClient(GenerationClient):
    """Test double that returns predictable generated text."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_index = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._responses:
            text = self._responses[self._call_index % len(self._responses)]
            self._call_index += 1
        else:
            text = "Generated mock response for testing."
        return GenerationResponse(text=text, model="mock-model")

    def generate_batch(self, requests: list[GenerationRequest]) -> BatchGenerationResponse:
        responses = [self.generate(r) for r in requests]
        return BatchGenerationResponse(responses=responses)

    def is_available(self) -> bool:
        return True


# ============================================
# Augmentation Planner Tests
# ============================================

class TestAugmentationPlanner:
    """Tests for the rule-based AugmentationPlanner."""

    def setup_method(self):
        self.planner = AugmentationPlanner()

    def test_plan_qa_dataset(self):
        metadata = {
            "task_detection": {"task": "qa", "confidence": 0.9},
            "profile": {"sample_count": 500, "quality_score": 0.7},
        }
        plan = self.planner.plan(metadata)
        assert plan.final_target > 500
        assert len(plan.strategies) >= 2
        strategy_names = [s.name for s in plan.strategies]
        assert "synthetic_qa" in strategy_names

    def test_plan_classification_dataset(self):
        metadata = {
            "task_detection": {"task": "classification", "confidence": 0.8},
            "profile": {"sample_count": 200, "quality_score": 0.6},
        }
        plan = self.planner.plan(metadata)
        assert plan.final_target > 200
        strategy_names = [s.name for s in plan.strategies]
        assert "paraphrase" in strategy_names

    def test_plan_high_quality_dataset(self):
        metadata = {
            "task_detection": {"task": "qa", "confidence": 0.9},
            "profile": {"sample_count": 1000, "quality_score": 0.9},
        }
        plan = self.planner.plan(metadata)
        # High quality = lower multiplier
        assert plan.final_target <= 1000 * 6

    def test_plan_empty_dataset(self):
        metadata = {
            "task_detection": {"task": "unknown"},
            "profile": {"sample_count": 0, "quality_score": 0.0},
        }
        plan = self.planner.plan(metadata)
        assert plan.final_target >= 100

    def test_plan_already_large_dataset(self):
        metadata = {
            "task_detection": {"task": "qa"},
            "profile": {"sample_count": 100000, "quality_score": 0.9},
        }
        plan = self.planner.plan(metadata)
        # Should not generate more since already above MAX_TARGET_SAMPLES
        assert len(plan.strategies) == 0

    def test_strategy_target_allocation(self):
        metadata = {
            "task_detection": {"task": "qa"},
            "profile": {"sample_count": 100, "quality_score": 0.5},
        }
        plan = self.planner.plan(metadata)
        total_allocated = sum(s.target_samples for s in plan.strategies)
        assert total_allocated > 0


# ============================================
# Strategy Tests
# ============================================

class TestParaphraseStrategy:
    def test_paraphrase_generates_samples(self):
        strategy = ParaphraseStrategy()
        client = MockGenerationClient(responses=["This is a paraphrased version."])
        samples = [{"text": "Original sample text."}]
        result = strategy.execute(samples, target_count=3, client=client)
        assert result.strategy_name == "paraphrase"
        assert result.generated_count == 3
        assert len(result.samples) == 3
        assert result.audit_trail["prompt_version"] == "paraphrase_v1"

    def test_paraphrase_empty_samples(self):
        strategy = ParaphraseStrategy()
        client = MockGenerationClient()
        result = strategy.execute([], target_count=5, client=client)
        assert result.generated_count == 0

    def test_paraphrase_name(self):
        assert ParaphraseStrategy().name == "paraphrase"


class TestQAGeneratorStrategy:
    def test_qa_generates_pairs(self):
        strategy = QAGeneratorStrategy()
        mock_qa = json.dumps({"question": "What is AI?", "answer": "Artificial Intelligence."})
        client = MockGenerationClient(responses=[mock_qa])
        samples = [{"question": "What is ML?", "answer": "Machine Learning."}]
        result = strategy.execute(samples, target_count=2, client=client)
        assert result.strategy_name == "synthetic_qa"
        assert result.generated_count == 2
        assert result.samples[0]["question"] == "What is AI?"

    def test_qa_handles_bad_json(self):
        strategy = QAGeneratorStrategy()
        client = MockGenerationClient(responses=["not json"])
        samples = [{"question": "Q?", "answer": "A."}]
        result = strategy.execute(samples, target_count=1, client=client)
        assert result.failed_count == 1
        assert result.generated_count == 0

    def test_qa_name(self):
        assert QAGeneratorStrategy().name == "synthetic_qa"


class TestSyntheticSampleStrategy:
    def test_synthetic_generates_samples(self):
        strategy = SyntheticSampleStrategy()
        mock_sample = json.dumps({"text": "New synthetic text", "label": "positive"})
        client = MockGenerationClient(responses=[mock_sample])
        samples = [{"text": "Original text", "label": "negative"}]
        result = strategy.execute(samples, target_count=2, client=client)
        assert result.strategy_name == "synthetic_sample"
        assert result.generated_count == 2

    def test_synthetic_name(self):
        assert SyntheticSampleStrategy().name == "synthetic_sample"


# ============================================
# Training Dataset Builder Tests
# ============================================

class TestTrainingDatasetBuilder:
    def setup_method(self):
        self.builder = TrainingDatasetBuilder()

    def test_build_alpaca_format(self):
        samples = [
            {"question": "What is AI?", "answer": "Artificial Intelligence."},
            {"instruction": "Explain ML", "output": "Machine Learning is..."},
        ]
        result = self.builder.build(samples, format_name="alpaca")
        assert len(result) == 2
        assert "instruction" in result[0]
        assert "output" in result[0]
        assert "input" in result[0]

    def test_build_chatml_format(self):
        samples = [
            {"question": "What is AI?", "answer": "Artificial Intelligence."},
        ]
        result = self.builder.build(samples, format_name="chatml")
        assert len(result) == 1
        assert "messages" in result[0]
        messages = result[0]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_build_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            self.builder.build([{"text": "test"}], format_name="xyz")

    def test_build_empty_samples(self):
        result = self.builder.build([], format_name="alpaca")
        assert result == []


# ============================================
# Export Service Tests
# ============================================

class TestExportService:
    def setup_method(self):
        self.service = ExportService()

    def test_export_json(self):
        records = [{"instruction": "Test", "input": "", "output": "Answer"}]
        data, content_type = self.service.export(records, file_format="json")
        assert content_type == "application/json"
        parsed = json.loads(data.decode("utf-8"))
        assert len(parsed) == 1

    def test_export_jsonl(self):
        records = [
            {"instruction": "Test1", "input": "", "output": "A1"},
            {"instruction": "Test2", "input": "", "output": "A2"},
        ]
        data, content_type = self.service.export(records, file_format="jsonl")
        assert content_type == "application/jsonl"
        lines = data.decode("utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_export_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.service.export([], file_format="xml")


# ============================================
# Quality Validator Tests (without model loading)
# ============================================

class TestQualityValidatorBasic:
    def test_empty_samples(self):
        validator = QualityValidator()
        accepted, rejected, report = validator.validate([], [])
        assert report.total_samples == 0
        assert accepted == []
        assert rejected == []

    def test_non_empty_returns_report(self):
        validator = QualityValidator()
        generated = [{"text": "Sample text for testing quality."}]
        original = [{"text": "Original text for comparison."}]
        accepted, rejected, report = validator.validate(generated, original)
        assert report.total_samples == 1
        assert len(accepted) + len(rejected) == 1


# ============================================
# Instruction Sanity Validator Tests
# ============================================

class TestInstructionSanityValidator:
    """Tests for the NLP-based instruction sanity checker."""

    def setup_method(self):
        from app.agents.augmentation_strategies.sanity_validator import (
            InstructionSanityValidator,
        )
        self.validator = InstructionSanityValidator()

    def test_valid_what_question(self):
        samples = [{"instruction": "What is quantum computing?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(valid) == 1
        assert len(rejected) == 0

    def test_valid_how_question(self):
        samples = [{"instruction": "How does gravity work?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(valid) == 1

    def test_valid_imperative(self):
        samples = [{"instruction": "Explain the theory of relativity."}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(valid) == 1

    def test_reject_who_inanimate(self):
        """'Who is quantum computing?' must be rejected."""
        samples = [{"instruction": "Who is quantum computing?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(rejected) == 1
        assert report.rejection_reasons.get("who_targets_inanimate", 0) >= 1

    def test_reject_who_computers(self):
        """'Who are computers?' must be rejected."""
        samples = [{"instruction": "Who are computers?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(rejected) == 1

    def test_reject_declarative_statement(self):
        """Declarative statements are not valid instructions."""
        samples = [{"instruction": "AI has become more advanced in recent years."}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(rejected) == 1
        assert "declarative_not_instruction" in report.rejection_reasons

    def test_reject_too_short(self):
        samples = [{"instruction": "Hi?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(rejected) == 1
        assert "too_short" in report.rejection_reasons

    def test_accept_who_person(self):
        """'Who is Albert Einstein?' should pass."""
        samples = [{"instruction": "Who is Albert Einstein?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(valid) == 1

    def test_mixed_batch(self):
        """Test a batch with both valid and invalid instructions."""
        samples = [
            {"instruction": "What is gravity?"},
            {"instruction": "Who is quantum computing?"},
            {"instruction": "Explain AI."},
            {"instruction": "AI is great."},
        ]
        valid, rejected, report = self.validator.validate(samples)
        assert report.total_checked == 4
        assert report.passed + report.failed == 4
        assert report.failed >= 2  # at least the two bad ones

    def test_rejected_examples_included(self):
        samples = [{"instruction": "Who is quantum computing?"}]
        valid, rejected, report = self.validator.validate(samples)
        assert len(report.rejected_examples) >= 1
        assert report.rejected_examples[0]["reason"] == "who_targets_inanimate"


# ============================================
# Validation Pipeline Tests
# ============================================

class TestValidationPipeline:
    """Tests for the Sanity → Quality → Dedup pipeline."""

    def test_empty_input(self):
        from app.agents.augmentation_strategies.validation_pipeline import (
            ValidationPipeline,
        )
        pipeline = ValidationPipeline()
        result, report = pipeline.run([], [])
        assert result == []
        assert report.total_input == 0

    def test_pipeline_filters_bad_instructions(self):
        from app.agents.augmentation_strategies.validation_pipeline import (
            ValidationPipeline,
        )
        pipeline = ValidationPipeline()
        generated = [
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "Who is quantum computing?", "output": "Something."},
        ]
        originals = [{"instruction": "What is ML?", "output": "Machine Learning."}]
        result, report = pipeline.run(generated, originals)
        # The bad instruction should be rejected
        assert report.rejected_samples >= 1
        assert "who_targets_inanimate" in report.rejection_reasons

    def test_pipeline_report_structure(self):
        from app.agents.augmentation_strategies.validation_pipeline import (
            ValidationPipeline,
        )
        pipeline = ValidationPipeline()
        generated = [{"text": "Valid text sample."}]
        originals = [{"text": "Original text."}]
        result, report = pipeline.run(generated, originals)
        assert hasattr(report, "sanity_report")
        assert hasattr(report, "quality_report")
        assert hasattr(report, "deduplication_report")
        assert hasattr(report, "rejection_reasons")
        assert hasattr(report, "rejected_examples")


# ============================================
# Deduplication Engine Tests (without model loading)
# ============================================

class TestDeduplicationEngineBasic:
    def test_empty_samples(self):
        engine = DeduplicationEngine()
        result, report = engine.deduplicate([])
        assert result == []
        assert report.total_input == 0

    def test_non_empty_returns_samples(self):
        engine = DeduplicationEngine()
        samples = [{"text": "Sample A"}, {"text": "Sample B"}]
        result, report = engine.deduplicate(samples)
        assert report.total_input == 2
