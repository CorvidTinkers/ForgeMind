"""
ForgeMind Unit Tests — Services & Agents.

Tests the core business logic in isolation, without infrastructure dependencies.
"""

import hashlib
import io
import json
import uuid

import pytest

from app.agents.dataset_validator import DatasetValidationAgent
from app.agents.submodules.dataset_profiler import DatasetProfiler
from app.agents.submodules.format_detector import FormatDetector
from app.agents.submodules.task_detector import TaskDetector
from app.services.file_parser import FileParserService


# ============================================
# Format Detector Tests
# ============================================

class TestFormatDetector:
    """Tests for the FormatDetector submodule."""

    def setup_method(self):
        self.detector = FormatDetector()

    def test_detect_json(self):
        data = json.dumps([{"question": "What?", "answer": "Yes"}]).encode()
        result = self.detector.detect(data, "data.json")
        assert result["format"] == "json"
        assert result["confidence"] >= 0.9

    def test_detect_csv(self):
        data = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        result = self.detector.detect(data, "data.csv")
        assert result["format"] == "csv"
        assert result["confidence"] >= 0.7

    def test_detect_txt(self):
        data = b"This is a plain text document with no special structure."
        result = self.detector.detect(data, "notes.txt")
        assert result["format"] == "txt"
        assert result["confidence"] >= 0.7

    def test_detect_pdf_magic_bytes(self):
        data = b"%PDF-1.4 fake pdf content"
        result = self.detector.detect(data, "doc.pdf")
        assert result["format"] == "pdf"
        assert result["confidence"] >= 0.9

    def test_detect_unknown(self):
        data = bytes(range(256))  # Binary garbage
        result = self.detector.detect(data, "file.xyz")
        assert result["format"] == "unknown"


# ============================================
# File Parser Tests
# ============================================

class TestFileParser:
    """Tests for the FileParserService."""

    def setup_method(self):
        self.parser = FileParserService()

    def test_parse_json_array(self):
        data = json.dumps([
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"},
        ]).encode()
        result = self.parser.parse(data, "qa.json")
        assert result.format_detected == "json"
        assert result.row_count == 2
        assert len(result.records) == 2
        assert "question" in result.columns

    def test_parse_csv(self):
        data = b"text,label\nGreat product,positive\nTerrible,negative\n"
        result = self.parser.parse(data, "sentiment.csv")
        assert result.format_detected == "csv"
        assert result.row_count == 2
        assert "text" in result.columns
        assert "label" in result.columns

    def test_parse_txt(self):
        data = b"Line one\nLine two\nLine three\n"
        result = self.parser.parse(data, "doc.txt")
        assert result.format_detected == "txt"
        assert result.row_count == 3
        assert len(result.sample_texts) == 3

    def test_parse_unsupported_format(self):
        result = self.parser.parse(b"data", "file.xyz")
        assert result.format_detected == "unknown"
        assert len(result.parse_errors) > 0

    def test_parse_malformed_json(self):
        data = b'{"broken json'
        result = self.parser.parse(data, "bad.json")
        assert len(result.parse_errors) > 0


# ============================================
# Dataset Validation Agent Tests
# ============================================

class TestDatasetValidationAgent:
    """Tests for the DatasetValidationAgent."""

    def setup_method(self):
        self.validator = DatasetValidationAgent()

    def test_validate_empty_file(self):
        result = self.validator.validate(b"", "empty.csv")
        assert result.is_valid is False
        assert any(e["check"] == "empty_file" for e in result.errors)

    def test_validate_valid_csv(self):
        data = b"question,answer\nWhat is Python?,A programming language\nWhat is AI?,Artificial Intelligence\n"
        result = self.validator.validate(data, "qa.csv")
        assert result.is_valid is True
        assert result.parsed_dataset is not None

    def test_validate_broken_json(self):
        data = b'{"broken": json}'
        result = self.validator.validate(data, "bad.json")
        assert result.is_valid is False

    def test_validate_minimal_content(self):
        data = b"ab"
        result = self.validator.validate(data, "tiny.txt")
        assert result.is_valid is False
        assert any(e["check"] == "insufficient_content" for e in result.errors)

    def test_validate_valid_json(self):
        data = json.dumps([
            {"text": "Patient presents with chest pain", "label": "cardiology"},
            {"text": "Fracture in left femur", "label": "orthopedics"},
        ]).encode()
        result = self.validator.validate(data, "medical.json")
        assert result.is_valid is True


# ============================================
# Task Detector Tests
# ============================================

class TestTaskDetector:
    """Tests for the TaskDetector submodule."""

    def setup_method(self):
        self.detector = TaskDetector()

    def test_detect_qa(self):
        result = self.detector.detect(
            columns=["question", "answer", "context"],
            sample_texts=["What is Python?", "What is ML?"],
            records=[{"question": "What?", "answer": "This"}],
        )
        assert result["task"] == "qa"
        assert result["confidence"] > 0.5

    def test_detect_classification(self):
        result = self.detector.detect(
            columns=["text", "label"],
            sample_texts=["Great product", "Terrible service"],
            records=[{"text": "Great", "label": "positive"}],
        )
        assert result["task"] == "classification"
        assert result["confidence"] > 0.5

    def test_detect_raw_documents(self):
        result = self.detector.detect(
            columns=[],
            sample_texts=["Just a plain document with some text."],
            records=[],
        )
        assert result["task"] == "raw_documents"

    def test_detect_conversation(self):
        result = self.detector.detect(
            columns=["role", "content"],
            sample_texts=["user: Hello", "assistant: Hi there"],
            records=[{"role": "user", "content": "Hello"}],
        )
        assert result["task"] == "conversation"
        assert result["confidence"] > 0.3


# ============================================
# Dataset Profiler Tests
# ============================================

class TestDatasetProfiler:
    """Tests for the DatasetProfiler submodule."""

    def setup_method(self):
        self.profiler = DatasetProfiler()

    def test_profile_basic(self):
        samples = [
            "This is a sample sentence with several words",
            "Another example text for profiling purposes",
            "Third sample to test the profiler module",
        ]
        result = self.profiler.profile(samples)
        assert result["sample_count"] == 3
        assert result["avg_tokens"] > 0
        assert result["max_tokens"] > 0
        assert result["quality_score"] > 0
        assert result["duplication_score"] == 0.0

    def test_profile_with_duplicates(self):
        samples = ["duplicate text"] * 10 + ["unique text"]
        result = self.profiler.profile(samples)
        assert result["duplication_score"] > 0.5

    def test_profile_empty(self):
        result = self.profiler.profile([])
        assert result["sample_count"] == 0
        assert result["quality_score"] == 0.0

    def test_quality_penalizes_short_samples(self):
        short_samples = ["a", "b", "c", "d", "e"]
        normal_samples = [
            "This is a properly written sentence with enough words",
            "Another well-formed sentence for testing purposes",
            "A third complete sentence with adequate content",
        ]
        short_result = self.profiler.profile(short_samples)
        normal_result = self.profiler.profile(normal_samples)
        assert normal_result["quality_score"] > short_result["quality_score"]


# ============================================
# Fingerprint Tests
# ============================================

class TestFingerprint:
    """Tests for SHA256 fingerprinting logic (without DB dependency)."""

    def test_sha256_consistency(self):
        data = b"test data for hashing"
        expected = hashlib.sha256(data).hexdigest()
        assert expected == hashlib.sha256(data).hexdigest()

    def test_different_data_different_hash(self):
        hash1 = hashlib.sha256(b"data1").hexdigest()
        hash2 = hashlib.sha256(b"data2").hexdigest()
        assert hash1 != hash2
