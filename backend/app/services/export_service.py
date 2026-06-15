"""
ForgeMind Dataset Export Service.

Handles exporting training datasets in downloadable formats:
- JSON (pretty-printed)
- JSONL (one record per line)

Supports Alpaca and ChatML formats via the TrainingDatasetBuilder.
"""

import io
import json
import logging

logger = logging.getLogger(__name__)


class ExportService:
    """
    Exports training datasets in downloadable file formats.

    Generates JSON or JSONL byte streams suitable for API responses
    or storage in MinIO.
    """

    def export_json(self, records: list[dict]) -> bytes:
        """
        Export records as a pretty-printed JSON file.

        Args:
            records: The training dataset records.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        content = json.dumps(records, indent=2, ensure_ascii=False)
        return content.encode("utf-8")

    def export_jsonl(self, records: list[dict]) -> bytes:
        """
        Export records as JSONL (one JSON object per line).

        Args:
            records: The training dataset records.

        Returns:
            UTF-8 encoded JSONL bytes.
        """
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        content = "\n".join(lines) + "\n"
        return content.encode("utf-8")

    def export(
        self,
        records: list[dict],
        file_format: str = "json",
    ) -> tuple[bytes, str]:
        """
        Export records in the specified file format.

        Args:
            records: The training dataset records.
            file_format: 'json' or 'jsonl'.

        Returns:
            Tuple of (file bytes, content type string).

        Raises:
            ValueError: If the file format is not supported.
        """
        if file_format == "json":
            data = self.export_json(records)
            content_type = "application/json"
        elif file_format == "jsonl":
            data = self.export_jsonl(records)
            content_type = "application/jsonl"
        else:
            raise ValueError(
                f"Unsupported export format: '{file_format}'. Supported: json, jsonl"
            )

        logger.info(
            "Exported %d records as %s (%d bytes)",
            len(records),
            file_format,
            len(data),
        )

        return data, content_type
