"""File-backed registry for local experiment matrix runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.experiments.matrix import ExperimentConfig


@dataclass(slots=True)
class ExperimentRecord:
    """Metadata for one local experiment run."""

    experiment_id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    run_ids: list[str]
    summary_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if the record is malformed."""

        _require_non_empty(self.experiment_id, "experiment_id")
        _require_non_empty(self.name, "name")
        if self.status not in {"created", "running", "completed", "completed_with_failures", "failed"}:
            raise ValueError("status is not a supported experiment status.")
        for field_name in ("total_runs", "completed_runs", "failed_runs"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if not isinstance(self.run_ids, list) or not all(
            isinstance(run_id, str) for run_id in self.run_ids
        ):
            raise ValueError("run_ids must be a list of strings.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record dictionary."""

        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_runs": self.total_runs,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "run_ids": list(self.run_ids),
            "summary_path": self.summary_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        """Create a record from a dictionary."""

        return cls(
            experiment_id=str(data["experiment_id"]),
            name=str(data["name"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            total_runs=int(data.get("total_runs", 0)),
            completed_runs=int(data.get("completed_runs", 0)),
            failed_runs=int(data.get("failed_runs", 0)),
            run_ids=list(data.get("run_ids", [])),
            summary_path=data.get("summary_path"),
            metadata=dict(data.get("metadata", {})),
        )


class ExperimentRegistry:
    """Local directory registry for experiment records and summaries."""

    def __init__(self, root: str | Path = "experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, config: ExperimentConfig) -> ExperimentRecord:
        """Create a registry directory and initial experiment record."""

        experiment_id = _make_experiment_id(config.name)
        experiment_dir = self.experiment_dir(experiment_id)
        (experiment_dir / "run_records").mkdir(parents=True, exist_ok=True)
        _write_json(experiment_dir / "experiment.json", config.to_dict())
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=config.name,
            status="created",
            created_at=_now(),
            updated_at=_now(),
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            run_ids=[],
            metadata={
                "description": config.description,
                "kind": config.kind,
                "tags": list(config.tags),
                **dict(config.metadata),
            },
        )
        self.save_record(record)
        return record

    def save_record(self, record: ExperimentRecord) -> ExperimentRecord:
        """Save an experiment record to ``record.json``."""

        record.updated_at = _now()
        record.validate()
        _write_json(self.experiment_dir(record.experiment_id) / "record.json", record.to_dict())
        return record

    def load_record(self, experiment_id: str) -> ExperimentRecord:
        """Load an experiment record by ID."""

        path = self.experiment_dir(experiment_id) / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment record does not exist: {experiment_id}")
        return ExperimentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_experiments(self) -> list[ExperimentRecord]:
        """List local experiment records sorted by creation time."""

        records = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "record.json"
            if record_path.exists():
                records.append(
                    ExperimentRecord.from_dict(
                        json.loads(record_path.read_text(encoding="utf-8"))
                    )
                )
        return sorted(records, key=lambda record: (record.created_at, record.experiment_id))

    def write_expanded_runs(self, experiment_id: str, runs) -> Path:
        """Write expanded child config envelopes."""

        path = self.experiment_dir(experiment_id) / "expanded_runs.json"
        return _write_json(path, [run.to_dict() for run in runs])

    def write_summary(self, experiment_id: str, summary: dict[str, Any]) -> Path:
        """Write experiment summary JSON."""

        path = self.experiment_dir(experiment_id) / "summary.json"
        return _write_json(path, summary)

    def write_summary_csv(self, experiment_id: str, rows: list[dict[str, Any]]) -> Path:
        """Write experiment summary rows as CSV."""

        path = self.experiment_dir(experiment_id) / "summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        preferred = [
            "experiment_id",
            "index",
            "kind",
            "status",
            "run_id",
            "mode",
            "model_type",
            "trainable_policy_mode",
            "final_train_loss",
            "final_eval_loss",
            "finite",
            "result_path",
            "error",
        ]
        keys = set()
        for row in rows:
            keys.update(row)
        fieldnames = [key for key in preferred if key in keys]
        fieldnames.extend(sorted(keys - set(fieldnames)))
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        return path

    def write_run_record(
        self,
        experiment_id: str,
        index: int,
        record: dict[str, Any],
    ) -> Path:
        """Write one child run record under ``run_records/``."""

        if type(index) is not int or index < 0:
            raise ValueError("index must be a non-negative integer.")
        path = self.experiment_dir(experiment_id) / "run_records" / f"{index}.json"
        return _write_json(path, record)

    def experiment_dir(self, experiment_id: str) -> Path:
        """Return the directory for an experiment ID."""

        return self.root / experiment_id


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _make_experiment_id(name: str) -> str:
    safe_name = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name or 'experiment'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
