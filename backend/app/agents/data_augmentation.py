"""
ForgeMind Data Augmentation Agent.

Orchestrator for the entire augmentation pipeline. Delegates to
the Augmentation Planner, executes strategies via the Strategy Pattern,
then runs the ValidationPipeline (Sanity → Quality → Dedup).

Phase 2.1: Uses ValidationPipeline abstraction; reports accepted/rejected
counts, rejection reasons, and a sample of rejected examples.
"""

import logging
from dataclasses import dataclass, field

from app.agents.augmentation_planner import AugmentationPlanner
from app.agents.augmentation_strategies.base import AugmentationStrategy
from app.agents.augmentation_strategies.paraphrase import ParaphraseStrategy
from app.agents.augmentation_strategies.qa_generator import QAGeneratorStrategy
from app.agents.augmentation_strategies.synthetic_sample import SyntheticSampleStrategy
from app.agents.augmentation_strategies.validation_pipeline import (
    ValidationPipeline,
    ValidationPipelineReport,
)
from app.infrastructure.generation.base import GenerationClient

logger = logging.getLogger(__name__)


# Strategy registry: maps strategy names to their implementations
STRATEGY_REGISTRY: dict[str, type[AugmentationStrategy]] = {
    "paraphrase": ParaphraseStrategy,
    "synthetic_qa": QAGeneratorStrategy,
    "synthetic_sample": SyntheticSampleStrategy,
}


@dataclass
class AugmentationReport:
    """
    Full report from an augmentation run.

    Provides traceability for every step of the augmentation pipeline.
    Phase 2.1: Adds accepted/rejected metrics, rejection reasons,
    and a small sample of rejected examples for debugging.
    """

    original_samples: int = 0
    generated_samples: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0
    duplicates_removed: int = 0
    final_samples: int = 0
    quality_score: float = 0.0
    strategies_used: list[dict] = field(default_factory=list)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejected_examples: list[dict] = field(default_factory=list)
    quality_report: dict = field(default_factory=dict)
    deduplication_report: dict = field(default_factory=dict)
    sanity_report: dict = field(default_factory=dict)
    audit_trails: list[dict] = field(default_factory=list)


class DataAugmentationAgent:
    """
    Orchestrator for dataset augmentation.

    Pipeline:
        1. Plan augmentation (Augmentation Planner)
        2. Execute strategies (Strategy Pattern)
        3. Validate & filter (ValidationPipeline: Sanity → Quality → Dedup)
        4. Return augmented samples + report
    """

    def __init__(self, generation_client: GenerationClient):
        self.client = generation_client
        self.planner = AugmentationPlanner()
        self.validation_pipeline = ValidationPipeline()

    def augment(
        self,
        samples: list[dict],
        metadata: dict,
    ) -> tuple[list[dict], AugmentationReport]:
        """
        Run the full augmentation pipeline.

        Args:
            samples: Original dataset samples (list of dicts).
            metadata: Phase 1 dataset intelligence metadata.

        Returns:
            Tuple of (final augmented samples, AugmentationReport).
        """
        original_count = len(samples)

        # Step 1: Plan
        plan = self.planner.plan(metadata)

        if not plan.strategies:
            logger.info("No augmentation needed. Returning original samples.")
            return samples, AugmentationReport(
                original_samples=original_count,
                final_samples=original_count,
            )

        # Step 2: Execute each strategy
        all_generated = []
        audit_trails = []
        strategies_used = []

        for allocation in plan.strategies:
            strategy_class = STRATEGY_REGISTRY.get(allocation.name)
            if strategy_class is None:
                logger.warning("Unknown strategy: %s. Skipping.", allocation.name)
                continue

            strategy = strategy_class()
            logger.info(
                "Executing strategy '%s' for %d samples",
                allocation.name,
                allocation.target_samples,
            )

            result = strategy.execute(
                samples=samples,
                target_count=allocation.target_samples,
                client=self.client,
                metadata=metadata,
            )

            all_generated.extend(result.samples)
            audit_trails.append(result.audit_trail)
            strategies_used.append({
                "name": allocation.name,
                "target_samples": allocation.target_samples,
                "generated_count": result.generated_count,
                "failed_count": result.failed_count,
            })

        # Step 3: Run ValidationPipeline (Sanity → Quality → Dedup)
        logger.info(
            "Running validation pipeline on %d generated samples", len(all_generated)
        )
        final_samples, pipeline_report = self.validation_pipeline.run(
            generated_samples=all_generated,
            original_samples=samples,
        )

        # Build report
        report = AugmentationReport(
            original_samples=original_count,
            generated_samples=len(all_generated),
            accepted_samples=pipeline_report.accepted_samples,
            rejected_samples=pipeline_report.rejected_samples,
            duplicates_removed=pipeline_report.deduplication_report.get(
                "duplicates_removed", 0
            ),
            final_samples=len(final_samples),
            quality_score=pipeline_report.quality_score,
            strategies_used=strategies_used,
            rejection_reasons=pipeline_report.rejection_reasons,
            rejected_examples=pipeline_report.rejected_examples,
            quality_report=pipeline_report.quality_report,
            deduplication_report=pipeline_report.deduplication_report,
            sanity_report=pipeline_report.sanity_report,
            audit_trails=audit_trails,
        )

        logger.info(
            "Augmentation complete: original=%d, generated=%d, "
            "accepted=%d, rejected=%d, final=%d, quality=%.2f",
            original_count,
            len(all_generated),
            pipeline_report.accepted_samples,
            pipeline_report.rejected_samples,
            len(final_samples),
            pipeline_report.quality_score,
        )

        return final_samples, report
