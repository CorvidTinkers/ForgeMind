"""
ForgeMind File Parser Service.

Extracts text content from supported file formats:
- PDF (via PyPDF2)
- TXT (plain text)
- CSV (via pandas)
- JSON (native)

Returns a unified ParsedDataset structure that downstream
analyzers consume regardless of original format.
"""

import csv
import io
import json
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
    logger.warning("PyPDF2 not installed. PDF parsing will be unavailable.")


@dataclass
class ParsedDataset:
    """
    Unified representation of a parsed dataset.

    Attributes:
        text_content: Extracted raw text (for language/domain detection).
        records: Structured records if applicable (for CSV/JSON).
        columns: Column names if tabular data.
        sample_texts: List of individual text samples for profiling.
        format_detected: The detected file format.
        row_count: Number of records/rows.
        parse_errors: Any non-fatal issues encountered during parsing.
    """

    text_content: str = ""
    records: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    sample_texts: list[str] = field(default_factory=list)
    format_detected: str = "unknown"
    row_count: int = 0
    parse_errors: list[str] = field(default_factory=list)


class FileParserService:
    """Parses uploaded files into a unified ParsedDataset."""

    def parse(self, data: bytes, filename: str) -> ParsedDataset:
        """
        Parse a file based on its extension.

        Args:
            data: Raw file bytes.
            filename: Original filename (used to detect extension).

        Returns:
            ParsedDataset with extracted content.
        """
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        parsers = {
            "pdf": self._parse_pdf,
            "txt": self._parse_txt,
            "csv": self._parse_csv,
            "json": self._parse_json,
        }

        parser = parsers.get(extension)
        if parser is None:
            logger.warning("Unsupported file format: %s", extension)
            return ParsedDataset(
                format_detected="unknown",
                parse_errors=[f"Unsupported file format: {extension}"],
            )

        try:
            result = parser(data)
            result.format_detected = extension
            return result
        except Exception as e:
            logger.error("Failed to parse %s: %s", filename, e)
            return ParsedDataset(
                format_detected=extension,
                parse_errors=[f"Parse error: {str(e)}"],
            )

    def _parse_pdf(self, data: bytes) -> ParsedDataset:
        """Extract text from PDF using PyPDF2."""
        if PdfReader is None:
            return ParsedDataset(parse_errors=["PyPDF2 not installed"])

        reader = PdfReader(io.BytesIO(data))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)
        return ParsedDataset(
            text_content=full_text,
            sample_texts=pages_text,
            row_count=len(pages_text),
        )

    def _parse_txt(self, data: bytes) -> ParsedDataset:
        """Parse plain text file."""
        text = data.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return ParsedDataset(
            text_content=text,
            sample_texts=lines,
            row_count=len(lines),
        )

    def _parse_csv(self, data: bytes) -> ParsedDataset:
        """Parse CSV file using pandas."""
        text = data.decode("utf-8", errors="replace")

        # Detect delimiter
        try:
            dialect = csv.Sniffer().sniff(text[:4096])
            sep = dialect.delimiter
        except csv.Error:
            sep = ","

        df = pd.read_csv(io.StringIO(text), sep=sep)
        columns = list(df.columns)

        # Build text samples from all string columns
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        sample_texts = []
        for _, row in df.iterrows():
            row_text = " ".join(
                str(row[col]) for col in text_cols if pd.notna(row[col])
            )
            if row_text.strip():
                sample_texts.append(row_text.strip())

        full_text = "\n".join(sample_texts)
        records = df.to_dict(orient="records")

        return ParsedDataset(
            text_content=full_text,
            records=records,
            columns=columns,
            sample_texts=sample_texts,
            row_count=len(df),
        )

    def _parse_json(self, data: bytes) -> ParsedDataset:
        """Parse JSON file (array of objects or single object)."""
        text = data.decode("utf-8", errors="replace")
        parsed = json.loads(text)

        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            # Could be a single record or a nested structure
            if any(isinstance(v, list) for v in parsed.values()):
                # Nested: find the first list and treat as records
                for key, value in parsed.items():
                    if isinstance(value, list) and len(value) > 0:
                        records = value
                        break
                else:
                    records = [parsed]
            else:
                records = [parsed]
        else:
            records = []

        # Extract columns from first record
        columns = list(records[0].keys()) if records else []

        # Build text samples
        sample_texts = []
        for record in records:
            parts = []
            for v in record.values():
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            text_sample = " ".join(parts)
            if text_sample:
                sample_texts.append(text_sample)

        full_text = "\n".join(sample_texts)

        return ParsedDataset(
            text_content=full_text,
            records=records,
            columns=columns,
            sample_texts=sample_texts,
            row_count=len(records),
        )
