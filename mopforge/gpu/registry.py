"""Local registry for GPUTrainer runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GPURunRecord:
    run_id: str
    name: str
    status: str
    output_dir: str
    created_at: str
    updated_at: str
    latest_checkpoint_path: str | None = None
    metrics_path: str | None = None
    result_path: str | None = None
    runtime_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPURunRecord":
        return cls(
            run_id=str(data["run_id"]),
            name=str(data.get("name", "")),
            status=str(data.get("status", "")),
            output_dir=str(data.get("output_dir", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            latest_checkpoint_path=data.get("latest_checkpoint_path"),
            metrics_path=data.get("metrics_path"),
            result_path=data.get("result_path"),
            runtime_path=data.get("runtime_path"),
            metadata=dict(data.get("metadata", {})),
        )


class GPURunRegistry:
    """File-backed GPU run registry rooted at ``gpu_runs``."""

    def __init__(self, root: str | Path = "gpu_runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        if not self.registry_path.exists():
            _write_json(self.registry_path, {"runs": []})

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def save_record(self, record: GPURunRecord) -> GPURunRecord:
        record_path = self.run_dir(record.run_id) / "record.json"
        _write_json(record_path, record.to_dict())
        registry = _read_json(self.registry_path)
        runs = [item for item in registry.get("runs", []) if item.get("run_id") != record.run_id]
        runs.append(record.to_dict())
        registry["runs"] = sorted(runs, key=lambda item: item.get("created_at", ""))
        _write_json(self.registry_path, registry)
        return record

    def list_runs(self) -> list[GPURunRecord]:
        records = [GPURunRecord.from_dict(item) for item in _read_json(self.registry_path).get("runs", [])]
        return sorted(records, key=lambda record: (record.created_at, record.run_id))

    def load_record(self, run_id: str) -> GPURunRecord:
        path = self.run_dir(run_id) / "record.json"
        if not path.exists():
            for record in self.list_runs():
                if record.run_id == run_id:
                    return record
            raise FileNotFoundError(
                f"GPU run does not exist: {run_id}. "
                "Run `mopforge gpu list` to see available local GPU runs."
            )
        return GPURunRecord.from_dict(_read_json(path))

    def latest_checkpoint(self, run_id: str) -> str:
        record = self.load_record(run_id)
        if record.latest_checkpoint_path:
            return record.latest_checkpoint_path
        checkpoints = sorted((self.run_dir(run_id) / "checkpoints").glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No GPU checkpoints for run: {run_id}. "
                "Run `mopforge gpu show <run_id>` to inspect the run record."
            )
        return str(checkpoints[-1])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
