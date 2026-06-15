"""
ForgeMind Dataset Validation Agent.

Runs BEFORE fingerprinting and analysis to ensure data integrity.

Checks:
- File corruption (unreadable bytes)
- Empty datasets
- Missing values in structured data
- Encoding issues
- Broken structures (malformed JSON, invalid CSV)
- Minimum content thresholds
"""

import logging
from dataclasses import dataclass, field

from app.services.file_parser import FileParserService, ParsedDataset

logger = logging.getLogger(__name__)

# Minimum number of characters to consider a dataset non-empty.
MIN_CONTENT_LENGTH = 10
# Maximum ratio of missing values before flagging quality issue.
MAX_MISSING_RATIO = 0.5


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool = True
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    parsed_dataset: ParsedDataset | None = None


class DatasetValidationAgent:
    """
    Validates dataset integrity before analysis.

    Returns a ValidationResult indicating whether the dataset
    can proceed to fingerprinting and analysis.
    """

    def __init__(self) -> None:
        self.parser = FileParserService()

    def validate(self, data: bytes, filename: str) -> ValidationResult:
        """
        Run all validation checks on the uploaded file.

        Args:
            data: Raw file bytes.
            filename: Original filename.

        Returns:
            ValidationResult with errors, warnings, and parsed data.
        """
        result = ValidationResult()

        # 1. Check if file is empty
        if not data or len(data) == 0:
            result.is_valid = False
            result.errors.append({
                "check": "empty_file",
                "message": "Uploaded file is empty (0 bytes).",
            })
            return result

        # 2. Attempt parsing
        parsed = self.parser.parse(data, filename)
        result.parsed_dataset = parsed

        # 3. Check for parse errors
        if parsed.parse_errors:
            for error in parsed.parse_errors:
                result.errors.append({
                    "check": "parse_error",
                    "message": error,
                })
            result.is_valid = False
            return result

        # 4. Check content length
        if len(parsed.text_content.strip()) < MIN_CONTENT_LENGTH:
            result.is_valid = False
            result.errors.append({
                "check": "insufficient_content",
                "message": (
                    f"Dataset has less than {MIN_CONTENT_LENGTH} characters "
                    f"of content after parsing."
                ),
            })
            return result

        # 5. Check row count
        if parsed.row_count == 0:
            result.is_valid = False
            result.errors.append({
                "check": "no_records",
                "message": "No records/rows found in the dataset.",
            })
            return result

        # 6. Structured data checks (CSV/JSON)
        if parsed.records:
            self._validate_structured(parsed, result)

        # 7. Encoding check (look for replacement characters)
        replacement_count = parsed.text_content.count("\ufffd")
        if replacement_count > 0:
            ratio = replacement_count / max(len(parsed.text_content), 1)
            if ratio > 0.05:
                result.warnings.append({
                    "check": "encoding_issues",
                    "message": (
                        f"Found {replacement_count} encoding replacement "
                        f"characters ({ratio:.1%} of content)."
                    ),
                })

        logger.info(
            "Validation complete for %s: valid=%s, errors=%d, warnings=%d",
            filename,
            result.is_valid,
            len(result.errors),
            len(result.warnings),
        )
        return result

    def _validate_structured(
        self,
        parsed: ParsedDataset,
        result: ValidationResult,
    ) -> None:
        """Run validation checks specific to structured (CSV/JSON) data."""
        if not parsed.records:
            return

        total_cells = 0
        missing_cells = 0

        for record in parsed.records:
            for key, value in record.items():
                total_cells += 1
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_cells += 1

        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            if missing_ratio > MAX_MISSING_RATIO:
                result.warnings.append({
                    "check": "high_missing_values",
                    "message": (
                        f"High ratio of missing/empty values: "
                        f"{missing_ratio:.1%} ({missing_cells}/{total_cells})."
                    ),
                })

        # Check for duplicate records
        if len(parsed.records) > 1:
            try:
                unique_texts = set(parsed.sample_texts)
                dup_ratio = 1.0 - (len(unique_texts) / len(parsed.sample_texts))
                if dup_ratio > 0.3:
                    result.warnings.append({
                        "check": "high_duplication",
                        "message": (
                            f"High duplication ratio: {dup_ratio:.1%} "
                            f"of records appear to be duplicates."
                        ),
                    })
            except Exception:
                pass
