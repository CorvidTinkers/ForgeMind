"""
ForgeMind Dataset Analyzer Agent.

The core intelligence orchestrator. Coordinates all submodules to produce
a comprehensive dataset intelligence report:

1. Format Detection
2. Language Detection
3. Domain Detection (embedding-based)
4. Task Detection
5. Dataset Profiling
6. Knowledge Extraction

Outputs a DatasetMetadata dict stored as JSONB in PostgreSQL.
"""

import logging
from typing import Any

import numpy as np

from app.agents.submodules.dataset_profiler import DatasetProfiler
from app.agents.submodules.domain_detector import DomainDetector
from app.agents.submodules.format_detector import FormatDetector
from app.agents.submodules.knowledge_extractor import KnowledgeExtractor
from app.agents.submodules.language_detector import LanguageDetector
from app.agents.submodules.task_detector import TaskDetector
from app.services.file_parser import ParsedDataset

logger = logging.getLogger(__name__)


class DatasetAnalyzerAgent:
    """
    Orchestrates the dataset analysis pipeline.

    Each submodule is injected and can be replaced independently.
    This allows future upgrades (e.g., swapping heuristic detectors
    with ML-based ones) without changing the orchestrator.
    """

    def __init__(
        self,
        format_detector: FormatDetector | None = None,
        language_detector: LanguageDetector | None = None,
        domain_detector: DomainDetector | None = None,
        task_detector: TaskDetector | None = None,
        profiler: DatasetProfiler | None = None,
        knowledge_extractor: KnowledgeExtractor | None = None,
    ) -> None:
        self.format_detector = format_detector or FormatDetector()
        self.language_detector = language_detector or LanguageDetector()
        self.domain_detector = domain_detector or DomainDetector()
        self.task_detector = task_detector or TaskDetector()
        self.profiler = profiler or DatasetProfiler()
        self.knowledge_extractor = knowledge_extractor or KnowledgeExtractor()

    def analyze(
        self,
        raw_data: bytes,
        filename: str,
        parsed: ParsedDataset,
    ) -> dict[str, Any]:
        """
        Run the full analysis pipeline on a parsed dataset.

        Args:
            raw_data: Raw file bytes (for format detection).
            filename: Original filename.
            parsed: Pre-parsed dataset from the validation stage.

        Returns:
            Complete metadata dict ready for JSONB storage.
        """
        logger.info("Starting dataset analysis for: %s", filename)

        # 1. Format Detection
        format_result = self.format_detector.detect(raw_data, filename)
        logger.info("Format: %s (confidence=%.2f)", format_result["format"], format_result["confidence"])

        # 2. Language Detection
        language_result = self.language_detector.detect(parsed.text_content)
        logger.info("Language: %s (confidence=%.2f)", language_result["language"], language_result["confidence"])

        # 3. Language Distribution
        language_distribution = self.language_detector.detect_distribution(
            parsed.sample_texts[:100]
        )

        # 4. Domain Detection (embedding-based)
        domain_result = self.domain_detector.detect(parsed.sample_texts)
        top_domains = domain_result.get("top_domains", [])
        dataset_embedding = domain_result.get("dataset_embedding")
        logger.info(
            "Top domain: %s",
            top_domains[0] if top_domains else "unknown",
        )

        # 5. Task Detection
        task_result = self.task_detector.detect(
            columns=parsed.columns,
            sample_texts=parsed.sample_texts,
            records=parsed.records,
        )
        logger.info("Task: %s (confidence=%.2f)", task_result["task"], task_result["confidence"])

        # 6. Dataset Profiling
        profile = self.profiler.profile(
            sample_texts=parsed.sample_texts,
            language_distribution=language_distribution,
        )

        # 7. Knowledge Extraction
        knowledge = self.knowledge_extractor.extract(parsed.sample_texts)
        logger.info(
            "Knowledge: %d entities, %d keywords, %d relationships",
            len(knowledge.get("entities", [])),
            len(knowledge.get("keywords", [])),
            len(knowledge.get("relationships", [])),
        )

        # Assemble the complete metadata
        metadata = {
            "format_detection": format_result,
            "language": language_result,
            "top_domains": top_domains,
            "task_detection": task_result,
            "profile": profile,
            "knowledge": knowledge,
        }

        logger.info("Dataset analysis complete for: %s", filename)

        return metadata, dataset_embedding
