"""
ForgeMind Validation Pipeline.

Orchestrates the full validation chain in a strict order:

    1. Instruction Sanity Validator  (NLP-based, fast)
    2. Quality Validator             (Embedding-based, filters low similarity)
    3. Deduplication Engine           (Embedding-based, removes near-dupes)

Each stage produces its own sub-report. The pipeline merges them into
a single ValidationPipelineReport that includes:
  - accepted / rejected counts
  - rejection reasons (aggregated from all stages)
  - a small sample of rejected examples for debugging
"""

import logging
from dataclasses import dataclass, field

from app.agents.augmentation_strategies.deduplication_engine import (
    DeduplicationEngine,
    DeduplicationReport,
)
from app.agents.augmentation_strategies.quality_validator import (
    QualityReport,
    QualityValidator,
)
from app.agents.augmentation_strategies.sanity_validator import (
    InstructionSanityValidator,
    SanityReport,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationPipelineReport:
    """
    Merged report from every validation stage.

    Fields mirror what the API will expose under the augmentation report.
    """

    # Totals
    total_input: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0

    # Aggregated rejection reasons across all stages
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    # A small selection of rejected samples for debugging
    rejected_examples: list[dict] = field(default_factory=list)

    # Sub-reports for full transparency
    sanity_report: dict = field(default_factory=dict)
    quality_report: dict = field(default_factory=dict)
    deduplication_report: dict = field(default_factory=dict)

    # Overall quality score (from QualityValidator)
    quality_score: float = 0.0

    # How many rejected examples to keep (across all stages combined)
    MAX_REJECTED_EXAMPLES: int = field(default=15, repr=False)


class ValidationPipeline:
    """
    Runs the full validation chain: Sanity → Quality → Deduplication.

    Usage::

        pipeline = ValidationPipeline()
        final_samples, report = pipeline.run(generated, originals)
    """

    def __init__(
        self,
        sanity_validator: InstructionSanityValidator | None = None,
        quality_validator: QualityValidator | None = None,
        deduplication_engine: DeduplicationEngine | None = None,
    ):
        self.sanity_validator = sanity_validator or InstructionSanityValidator()
        self.quality_validator = quality_validator or QualityValidator()
        self.deduplication_engine = deduplication_engine or DeduplicationEngine()

    def run(
        self,
        generated_samples: list[dict],
        original_samples: list[dict],
    ) -> tuple[list[dict], ValidationPipelineReport]:
        """
        Execute the full validation pipeline.

        Args:
            generated_samples: Raw output from augmentation strategies.
            original_samples: The original dataset samples.

        Returns:
            Tuple of (final_accepted_samples, ValidationPipelineReport).
        """
        total_input = len(generated_samples)
        all_rejection_reasons: dict[str, int] = {}
        all_rejected_examples: list[dict] = []

        # ----------------------------------------------------------
        # Stage 1: Instruction Sanity
        # ----------------------------------------------------------
        logger.info("Validation Stage 1/3: Instruction Sanity (%d samples)", len(generated_samples))
        sane_samples, sanity_rejected, sanity_report = self.sanity_validator.validate(
            generated_samples
        )
        self._merge_rejections(
            all_rejection_reasons,
            all_rejected_examples,
            sanity_report.rejection_reasons,
            sanity_report.rejected_examples,
        )

        # ----------------------------------------------------------
        # Stage 2: Quality Validation (embedding similarity filter)
        # ----------------------------------------------------------
        logger.info("Validation Stage 2/3: Quality Validation (%d samples)", len(sane_samples))
        quality_accepted, quality_rejected, quality_report = self.quality_validator.validate(
            generated_samples=sane_samples,
            original_samples=original_samples,
        )
        self._merge_rejections(
            all_rejection_reasons,
            all_rejected_examples,
            quality_report.rejection_reasons,
            quality_report.rejected_examples,
        )

        # ----------------------------------------------------------
        # Stage 3: Deduplication
        # ----------------------------------------------------------
        logger.info("Validation Stage 3/3: Deduplication (%d samples)", len(quality_accepted))
        deduplicated, dedup_report = self.deduplication_engine.deduplicate(
            generated_samples=quality_accepted,
            original_samples=original_samples,
        )

        # Dedup removals count as rejections too
        dedup_rejected_count = dedup_report.duplicates_removed
        if dedup_rejected_count > 0:
            all_rejection_reasons["duplicate"] = (
                all_rejection_reasons.get("duplicate", 0) + dedup_rejected_count
            )

        # ----------------------------------------------------------
        # Build merged report
        # ----------------------------------------------------------
        total_rejected = total_input - len(deduplicated)

        pipeline_report = ValidationPipelineReport(
            total_input=total_input,
            accepted_samples=len(deduplicated),
            rejected_samples=total_rejected,
            rejection_reasons=all_rejection_reasons,
            rejected_examples=all_rejected_examples[:ValidationPipelineReport.MAX_REJECTED_EXAMPLES],
            sanity_report={
                "total_checked": sanity_report.total_checked,
                "passed": sanity_report.passed,
                "failed": sanity_report.failed,
                "rejection_reasons": sanity_report.rejection_reasons,
            },
            quality_report={
                "semantic_similarity": quality_report.semantic_similarity,
                "diversity_score": quality_report.diversity_score,
                "duplicate_score": quality_report.duplicate_score,
                "quality_score": quality_report.quality_score,
                "accepted": quality_report.accepted_samples,
                "rejected": quality_report.rejected_samples,
            },
            deduplication_report={
                "total_input": dedup_report.total_input,
                "duplicates_removed": dedup_report.duplicates_removed,
                "final_count": dedup_report.final_count,
            },
            quality_score=quality_report.quality_score,
        )

        logger.info(
            "Validation pipeline complete: input=%d, accepted=%d, rejected=%d (sanity=%d, quality=%d, dedup=%d)",
            total_input,
            len(deduplicated),
            total_rejected,
            sanity_report.failed,
            quality_report.rejected_samples,
            dedup_rejected_count,
        )

        return deduplicated, pipeline_report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_rejections(
        target_reasons: dict[str, int],
        target_examples: list[dict],
        new_reasons: dict[str, int],
        new_examples: list[dict],
    ) -> None:
        """Merge rejection reasons and examples from a sub-report."""
        for reason, count in new_reasons.items():
            target_reasons[reason] = target_reasons.get(reason, 0) + count
        target_examples.extend(new_examples)
