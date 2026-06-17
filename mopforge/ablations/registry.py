"""File-backed ablation registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class AblationRecord:
    ablation_id: str
    name: str
    status: str
    created_at: str
    experiment_id: str | None = None
    analysis_id: str | None = None
    benchmark_ids: list[str] = field(default_factory=list)
    report_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_id": self.ablation_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "experiment_id": self.experiment_id,
            "analysis_id": self.analysis_id,
            "benchmark_ids": list(self.benchmark_ids),
            "report_path": self.report_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationRecord":
        return cls(
            ablation_id=str(data["ablation_id"]),
            name=str(data["name"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            experiment_id=data.get("experiment_id"),
            analysis_id=data.get("analysis_id"),
            benchmark_ids=list(data.get("benchmark_ids", [])),
            report_path=data.get("report_path"),
            metadata=dict(data.get("metadata", {})),
        )


class AblationRegistry:
    def __init__(self, root: str | Path = "ablations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str) -> AblationRecord:
        ablation_id = _id(name)
        directory = self.ablation_dir(ablation_id)
        directory.mkdir(parents=True, exist_ok=True)
        record = AblationRecord(ablation_id=ablation_id, name=name, status="created", created_at=_now())
        self.save(record)
        return record

    def save(self, record: AblationRecord) -> None:
        _write_json(self.ablation_dir(record.ablation_id) / "record.json", record.to_dict())

    def load(self, ablation_id: str) -> AblationRecord:
        path = self.ablation_dir(ablation_id) / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"Ablation record does not exist: {ablation_id}")
        return AblationRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[AblationRecord]:
        records = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "record.json"
            if record_path.exists():
                records.append(AblationRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8"))))
        return sorted(records, key=lambda item: (item.created_at, item.ablation_id))

    def ablation_dir(self, ablation_id: str) -> Path:
        return self.root / ablation_id


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _id(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe or 'ablation'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
