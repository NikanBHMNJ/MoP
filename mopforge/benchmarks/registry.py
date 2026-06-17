"""File-backed benchmark registry."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.benchmarks.config import BenchmarkConfig
from mopforge.benchmarks.metrics import flatten_metrics


@dataclass(slots=True)
class BenchmarkRecord:
    """Metadata for one benchmark run."""

    benchmark_id: str
    name: str
    benchmark_type: str
    status: str
    created_at: str
    completed_at: str | None = None
    metrics_path: str | None = None
    metrics_csv_path: str | None = None
    examples_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if this record is malformed."""

        for field_name in ("benchmark_id", "name", "benchmark_type", "status", "created_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.status not in {"created", "running", "completed", "failed"}:
            raise ValueError("status is not a supported benchmark status.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record dictionary."""

        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "benchmark_type": self.benchmark_type,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metrics_path": self.metrics_path,
            "metrics_csv_path": self.metrics_csv_path,
            "examples_path": self.examples_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecord":
        """Create a record from a dictionary."""

        return cls(
            benchmark_id=str(data["benchmark_id"]),
            name=str(data["name"]),
            benchmark_type=str(data["benchmark_type"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            completed_at=data.get("completed_at"),
            metrics_path=data.get("metrics_path"),
            metrics_csv_path=data.get("metrics_csv_path"),
            examples_path=data.get("examples_path"),
            metadata=dict(data.get("metadata", {})),
        )


class BenchmarkRegistry:
    """Local benchmark registry rooted at ``benchmarks/``."""

    def __init__(self, root: str | Path = "benchmarks") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_benchmark(self, config: BenchmarkConfig) -> BenchmarkRecord:
        """Create a benchmark directory and initial record."""

        benchmark_id = _make_benchmark_id(config.name)
        directory = self.benchmark_dir(benchmark_id)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "benchmark.json", config.to_dict())
        record = BenchmarkRecord(
            benchmark_id=benchmark_id,
            name=config.name,
            benchmark_type=config.benchmark_type,
            status="created",
            created_at=_now(),
            metadata=dict(config.metadata),
        )
        self.save_record(record)
        return record

    def save_record(self, record: BenchmarkRecord) -> BenchmarkRecord:
        """Write ``record.json``."""

        record.validate()
        _write_json(self.benchmark_dir(record.benchmark_id) / "record.json", record.to_dict())
        return record

    def load_record(self, benchmark_id: str) -> BenchmarkRecord:
        """Load one benchmark record by ID."""

        path = self.benchmark_dir(benchmark_id) / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark record does not exist: {benchmark_id}")
        return BenchmarkRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_benchmarks(self) -> list[BenchmarkRecord]:
        """List local benchmark records."""

        records = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "record.json"
            if record_path.exists():
                records.append(
                    BenchmarkRecord.from_dict(
                        json.loads(record_path.read_text(encoding="utf-8"))
                    )
                )
        return sorted(records, key=lambda record: (record.created_at, record.benchmark_id))

    def write_metrics(self, benchmark_id: str, metrics: dict[str, Any]) -> Path:
        """Write metrics JSON."""

        return _write_json(self.benchmark_dir(benchmark_id) / "metrics.json", metrics)

    def write_metrics_csv(self, benchmark_id: str, rows: list[dict[str, Any]]) -> Path:
        """Write flattened metrics CSV."""

        path = self.benchmark_dir(benchmark_id) / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        flattened = [flatten_metrics(row) for row in rows]
        keys = sorted({key for row in flattened for key in row})
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            for row in flattened:
                writer.writerow({key: row.get(key) for key in keys})
        return path

    def write_examples(self, benchmark_id: str, examples: list[dict[str, Any]]) -> Path:
        """Write benchmark example details."""

        return _write_json(self.benchmark_dir(benchmark_id) / "examples.json", examples)

    def benchmark_dir(self, benchmark_id: str) -> Path:
        """Return directory for one benchmark."""

        return self.root / benchmark_id


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _make_benchmark_id(name: str) -> str:
    safe_name = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name or 'benchmark'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
