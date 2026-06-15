"""
ForgeMind Data Augmentation Agent.

Orchestrator for the entire augmentation pipeline. Delegates to
the Augmentation Planner, executes strategies via the Strategy Pattern,
validates quality, deduplicates, and returns the final augmented dataset.
"""

import logging
from dataclasses import dataclass, field

from app.agents.augmentation_planner import AugmentationPlanner
from app.agents.augmentation_strategies.base import AugmentationStrategy
from app.agents.augmentation_strategies.deduplication_engine import (
    DeduplicationEngine,
    DeduplicationReport,
)
from app.agents.augmentation_strategies.paraphrase import ParaphraseStrategy
from app.agents.augmentation_strategies.qa_generator import QAGeneratorStrategy
from app.agents.augmentation_strategies.quality_validator import (
    QualityReport,
    QualityValidator,
)
from app.agents.augmentation_strategies.synthetic_sample import SyntheticSampleStrategy
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
    """

    original_samples: int = 0
    generated_samples: int = 0
    duplicates_removed: int = 0
    final_samples: int = 0
    quality_score: float = 0.0
    strategies_used: list[dict] = field(default_factory=list)
    quality_report: dict = field(default_factory=dict)
    deduplication_report: dict = field(default_factory=dict)
    audit_trails: list[dict] = field(default_factory=list)


class DataAugmentationAgent:
    """
    Orchestrator for dataset augmentation.

    Pipeline:
        1. Plan augmentation (Augmentation Planner)
        2. Execute strategies (Strategy Pattern)
        3. Validate quality (QualityValidator)
        4. Deduplicate (DeduplicationEngine)
        5. Return augmented samples + report
    """

    def __init__(self, generation_client: GenerationClient):
        self.client = generation_client
        self.planner = AugmentationPlanner()
        self.quality_validator = QualityValidator()
        self.deduplication_engine = DeduplicationEngine()

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

        # Step 3: Quality validation
        logger.info("Validating quality of %d generated samples", len(all_generated))
        quality_report = self.quality_validator.validate(
            generated_samples=all_generated,
            original_samples=samples,
        )

        # Step 4: Deduplication
        logger.info("Running deduplication on %d samples", len(all_generated))
        deduplicated, dedup_report = self.deduplication_engine.deduplicate(
            generated_samples=all_generated,
            original_samples=samples,
        )

        # Build report
        report = AugmentationReport(
            original_samples=original_count,
            generated_samples=len(all_generated),
            duplicates_removed=dedup_report.duplicates_removed,
            final_samples=len(deduplicated),
            quality_score=quality_report.quality_score,
            strategies_used=strategies_used,
            quality_report={
                "semantic_similarity": quality_report.semantic_similarity,
                "diversity_score": quality_report.diversity_score,
                "duplicate_score": quality_report.duplicate_score,
                "quality_score": quality_report.quality_score,
                "flagged_low_quality": quality_report.flagged_low_quality,
            },
            deduplication_report={
                "total_input": dedup_report.total_input,
                "duplicates_removed": dedup_report.duplicates_removed,
                "final_count": dedup_report.final_count,
            },
            audit_trails=audit_trails,
        )

        logger.info(
            "Augmentation complete: original=%d, generated=%d, deduped=%d, final=%d, quality=%.2f",
            original_count,
            len(all_generated),
            dedup_report.duplicates_removed,
            len(deduplicated),
            quality_report.quality_score,
        )

        return deduplicated, report
