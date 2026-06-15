"""
ForgeMind Augmentation Planner.

Rule-based planner that determines the optimal augmentation strategy
before any generation begins. Analyzes Phase 1 dataset intelligence
metadata to decide which generators to invoke and how many samples
each should produce.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Default target multiplier: aim for 10x the original dataset size
DEFAULT_TARGET_MULTIPLIER = 10
# Minimum final target to bother augmenting
MIN_TARGET_SAMPLES = 100
# Maximum single-run generation cap to prevent runaway costs
MAX_TARGET_SAMPLES = 50000


@dataclass
class StrategyAllocation:
    """A single augmentation strategy with its target sample count."""

    name: str
    target_samples: int


@dataclass
class AugmentationPlan:
    """
    The output of the Augmentation Planner.

    Contains the list of strategies to execute and the overall
    final target sample count.
    """

    strategies: list[StrategyAllocation] = field(default_factory=list)
    final_target: int = 0


class AugmentationPlanner:
    """
    Rule-based planner for dataset augmentation.

    Reads the dataset metadata (type, domain, sample count, quality)
    from Phase 1 and produces an AugmentationPlan.
    """

    def plan(self, metadata: dict) -> AugmentationPlan:
        """
        Generate an augmentation plan from dataset metadata.

        Args:
            metadata: Phase 1 dataset intelligence metadata dict.
                Expected keys: task_detection, top_domains, profile.

        Returns:
            AugmentationPlan with strategies and target counts.
        """
        task_info = metadata.get("task_detection", {})
        task_type = task_info.get("task", "unknown")
        profile = metadata.get("profile", {})
        sample_count = profile.get("sample_count", 0)
        quality_score = profile.get("quality_score", 0.5)

        # Calculate final target
        final_target = self._calculate_target(sample_count, quality_score)

        # Determine strategies based on dataset task type
        samples_needed = final_target - sample_count
        if samples_needed <= 0:
            logger.info(
                "Dataset already meets target (%d >= %d). No augmentation needed.",
                sample_count,
                final_target,
            )
            return AugmentationPlan(strategies=[], final_target=sample_count)

        strategies = self._select_strategies(task_type, samples_needed)

        plan = AugmentationPlan(strategies=strategies, final_target=final_target)

        logger.info(
            "Augmentation plan: task=%s, original=%d, target=%d, strategies=%s",
            task_type,
            sample_count,
            final_target,
            [(s.name, s.target_samples) for s in strategies],
        )

        return plan

    def _calculate_target(self, sample_count: int, quality_score: float) -> int:
        """
        Calculate the target sample count.

        Higher quality datasets get a lower multiplier (less augmentation needed).
        Lower quality datasets get more aggressive augmentation.
        """
        if sample_count == 0:
            return MIN_TARGET_SAMPLES

        # Scale multiplier inversely with quality
        if quality_score >= 0.8:
            multiplier = 5
        elif quality_score >= 0.5:
            multiplier = DEFAULT_TARGET_MULTIPLIER
        else:
            multiplier = 15

        target = sample_count * multiplier
        return min(max(target, MIN_TARGET_SAMPLES), MAX_TARGET_SAMPLES)

    def _select_strategies(
        self, task_type: str, samples_needed: int
    ) -> list[StrategyAllocation]:
        """
        Select augmentation strategies and allocate samples based on task type.

        Each task type gets a different ratio of strategies.
        """
        # Strategy ratios per task type
        ratios = {
            "qa": [
                ("paraphrase", 0.30),
                ("synthetic_qa", 0.70),
            ],
            "classification": [
                ("paraphrase", 0.50),
                ("synthetic_sample", 0.50),
            ],
            "conversation": [
                ("paraphrase", 0.30),
                ("synthetic_sample", 0.70),
            ],
            "summarization": [
                ("paraphrase", 0.60),
                ("synthetic_sample", 0.40),
            ],
            "raw_documents": [
                ("paraphrase", 0.40),
                ("synthetic_qa", 0.40),
                ("synthetic_sample", 0.20),
            ],
        }

        selected_ratios = ratios.get(task_type, ratios["raw_documents"])

        strategies = []
        for strategy_name, ratio in selected_ratios:
            target = max(1, int(samples_needed * ratio))
            strategies.append(
                StrategyAllocation(name=strategy_name, target_samples=target)
            )

        return strategies
