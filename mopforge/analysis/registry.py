"""File-backed registry for local analysis reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.analysis.config import AnalysisConfig
from mopforge.benchmarks.metrics import flatten_metrics


@dataclass(slots=True)
class AnalysisRecord:
    """Metadata for one local analysis report."""

    analysis_id: str
    name: str
    status: str
    created_at: str
    completed_at: str | None = None
    report_path: str | None = None
    normalized_results_path: str | None = None
    comparison_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if the record is malformed."""

        for field_name in ("analysis_id", "name", "status", "created_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.status not in {"created", "running", "completed", "failed"}:
            raise ValueError("status is not a supported analysis status.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record dictionary."""

        return {
            "analysis_id": self.analysis_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "report_path": self.report_path,
            "normalized_results_path": self.normalized_results_path,
            "comparison_path": self.comparison_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRecord":
        """Create a record from a dictionary."""

        return cls(
            analysis_id=str(data["analysis_id"]),
            name=str(data["name"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            completed_at=data.get("completed_at"),
            report_path=data.get("report_path"),
            normalized_results_path=data.get("normalized_results_path"),
            comparison_path=data.get("comparison_path"),
            metadata=dict(data.get("metadata", {})),
        )


class AnalysisRegistry:
    """Local report registry rooted at ``reports/``."""

    def __init__(self, root: str | Path = "reports") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_analysis(self, config: AnalysisConfig) -> AnalysisRecord:
        """Create a report directory and initial record."""

        analysis_id = _make_analysis_id(config.name)
        directory = self.analysis_dir(analysis_id)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "analysis.json", config.to_dict())
        record = AnalysisRecord(
            analysis_id=analysis_id,
            name=config.name,
            status="created",
            created_at=_now(),
            metadata={
                "description": config.description,
                "experiment_ids": list(config.experiment_ids),
                "benchmark_ids": list(config.benchmark_ids),
                "run_paths": list(config.run_paths),
                **dict(config.metadata),
            },
        )
        self.save_record(record)
        return record

    def save_record(self, record: AnalysisRecord) -> AnalysisRecord:
        """Write ``record.json``."""

        record.validate()
        _write_json(self.analysis_dir(record.analysis_id) / "record.json", record.to_dict())
        return record

    def load_record(self, analysis_id: str) -> AnalysisRecord:
        """Load one analysis record by ID."""

        path = self.analysis_dir(analysis_id) / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"Analysis record does not exist: {analysis_id}")
        return AnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_analyses(self) -> list[AnalysisRecord]:
        """List local analysis records sorted by creation time."""

        records: list[AnalysisRecord] = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "record.json"
            if record_path.exists():
                records.append(
                    AnalysisRecord.from_dict(
                        json.loads(record_path.read_text(encoding="utf-8"))
                    )
                )
        return sorted(records, key=lambda record: (record.created_at, record.analysis_id))

    def write_normalized_results(
        self,
        analysis_id: str,
        rows: list[dict[str, Any]],
    ) -> Path:
        """Write normalized rows as JSON."""

        return _write_json(self.analysis_dir(analysis_id) / "normalized_results.json", rows)

    def write_normalized_results_csv(
        self,
        analysis_id: str,
        rows: list[dict[str, Any]],
    ) -> Path:
        """Write normalized rows as CSV."""

        return _write_csv(self.analysis_dir(analysis_id) / "normalized_results.csv", rows)

    def write_comparison(self, analysis_id: str, comparison: dict[str, Any]) -> Path:
        """Write comparison JSON."""

        return _write_json(self.analysis_dir(analysis_id) / "comparison.json", comparison)

    def write_comparison_csv(
        self,
        analysis_id: str,
        comparison: dict[str, Any],
    ) -> Path:
        """Write a practical flattened comparison CSV."""

        rows = []
        for group in comparison.get("grouped_summaries", []):
            rows.append({"section": "group", **dict(group)})
        for row in comparison.get("deltas_vs_baseline", []):
            rows.append({"section": "delta", **dict(row)})
        if comparison.get("best_row") is not None:
            rows.append({"section": "best_row", **dict(comparison["best_row"])})
        if not rows:
            rows.append(
                {
                    "section": "summary",
                    "row_count": comparison.get("row_count", 0),
                    "warnings": "; ".join(comparison.get("warnings", [])),
                }
            )
        return _write_csv(self.analysis_dir(analysis_id) / "comparison.csv", rows)

    def write_report_markdown(self, analysis_id: str, markdown: str) -> Path:
        """Write ``report.md``."""

        path = self.analysis_dir(analysis_id) / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        return path

    def analysis_dir(self, analysis_id: str) -> Path:
        """Return directory for one analysis ID."""

        return self.root / analysis_id


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_metrics(row) for row in rows]
    keys = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in flattened:
            writer.writerow({key: row.get(key) for key in keys})
    return path


def _make_analysis_id(name: str) -> str:
    safe_name = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name or 'analysis'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
