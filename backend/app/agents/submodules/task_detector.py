"""
ForgeMind Task Detector.

Detects the dataset type/task:
- QA Dataset
- Conversation Dataset
- Classification Dataset
- Summarization Dataset
- Raw Documents

Uses structural heuristics (column names, data patterns) combined
with content analysis for confident classification.
"""

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Column name patterns that indicate specific task types
QA_COLUMNS = {"question", "answer", "context", "q", "a", "query", "response", "prompt", "completion"}
CONVERSATION_COLUMNS = {"message", "reply", "turn", "speaker", "role", "content", "dialogue", "utterance", "user", "assistant", "system"}
CLASSIFICATION_COLUMNS = {"label", "class", "category", "tag", "sentiment", "target", "type"}
SUMMARIZATION_COLUMNS = {"summary", "abstract", "article", "document", "text", "source", "target", "highlights"}

# Content patterns
QA_PATTERNS = [
    r"\?\s*$",  # Ends with question mark
    r"^(what|who|where|when|why|how|which|is|are|do|does|can|could|would|should)\b",
]
CONVERSATION_PATTERNS = [
    r"^(user|assistant|human|ai|bot|agent|system)\s*:",
    r"^\[.*?\]\s*:",
]


class TaskDetector:
    """Detects the dataset task type using structural and content analysis."""

    def detect(
        self,
        columns: list[str],
        sample_texts: list[str],
        records: list[dict],
    ) -> dict:
        """
        Detect the dataset task type.

        Args:
            columns: Column names from the parsed dataset.
            sample_texts: List of text samples.
            records: Structured records (if available).

        Returns:
            Dict with 'task' and 'confidence' keys.
        """
        scores = {
            "qa": 0.0,
            "conversation": 0.0,
            "classification": 0.0,
            "summarization": 0.0,
            "raw_documents": 0.0,
        }

        # 1. Column-based detection (strongest signal)
        if columns:
            col_set = {c.lower().strip() for c in columns}
            scores["qa"] += self._column_overlap(col_set, QA_COLUMNS) * 0.6
            scores["conversation"] += self._column_overlap(col_set, CONVERSATION_COLUMNS) * 0.6
            scores["classification"] += self._column_overlap(col_set, CLASSIFICATION_COLUMNS) * 0.6
            scores["summarization"] += self._column_overlap(col_set, SUMMARIZATION_COLUMNS) * 0.6

        # 2. Content-based detection
        if sample_texts:
            content_scores = self._analyze_content(sample_texts)
            for task, score in content_scores.items():
                scores[task] += score * 0.4

        # 3. Structure-based detection
        if records:
            struct_scores = self._analyze_structure(records)
            for task, score in struct_scores.items():
                scores[task] = max(scores[task], score)

        # 4. If no strong signal, default to raw_documents
        max_score = max(scores.values())
        if max_score < 0.2:
            scores["raw_documents"] = 0.7

        # Find the best match
        best_task = max(scores, key=scores.get)
        confidence = min(round(scores[best_task], 4), 1.0)

        logger.info(
            "Task detection: %s (confidence=%.3f), all_scores=%s",
            best_task,
            confidence,
            {k: round(v, 3) for k, v in scores.items()},
        )

        return {"task": best_task, "confidence": confidence}

    def _column_overlap(self, columns: set[str], pattern_set: set[str]) -> float:
        """Calculate overlap ratio between column names and pattern set."""
        if not columns:
            return 0.0
        overlap = columns & pattern_set
        return len(overlap) / max(len(pattern_set), 1)

    def _analyze_content(self, sample_texts: list[str]) -> dict[str, float]:
        """Analyze text content for task-type patterns."""
        scores = {
            "qa": 0.0,
            "conversation": 0.0,
            "classification": 0.0,
            "summarization": 0.0,
            "raw_documents": 0.0,
        }

        total = len(sample_texts)
        if total == 0:
            return scores

        qa_count = 0
        conv_count = 0
        long_text_count = 0

        for text in sample_texts[:200]:
            text_lower = text.lower().strip()

            # QA patterns
            for pattern in QA_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    qa_count += 1
                    break

            # Conversation patterns
            for pattern in CONVERSATION_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    conv_count += 1
                    break

            # Long texts suggest summarization or raw documents
            if len(text) > 500:
                long_text_count += 1

        analyzed = min(total, 200)
        scores["qa"] = qa_count / analyzed if analyzed > 0 else 0
        scores["conversation"] = conv_count / analyzed if analyzed > 0 else 0

        if long_text_count / analyzed > 0.5:
            scores["summarization"] = 0.3
            scores["raw_documents"] = 0.4

        return scores

    def _analyze_structure(self, records: list[dict]) -> dict[str, float]:
        """Analyze record structure for task-type indicators."""
        scores = {
            "qa": 0.0,
            "conversation": 0.0,
            "classification": 0.0,
            "summarization": 0.0,
            "raw_documents": 0.0,
        }

        if not records:
            return scores

        sample = records[0]

        # Check for common QA structures
        keys_lower = {k.lower() for k in sample.keys()}
        if {"question", "answer"} <= keys_lower or {"prompt", "completion"} <= keys_lower:
            scores["qa"] = 0.85

        # Check for conversation structures (list of turns)
        for key, value in sample.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and "role" in value[0]:
                    scores["conversation"] = 0.90
                    break

        # Check for classification (short text + label)
        if "label" in keys_lower or "class" in keys_lower:
            scores["classification"] = 0.85

        return scores
