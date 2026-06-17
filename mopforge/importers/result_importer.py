"""Local filesystem result importer."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.analysis.loading import (
    load_benchmark_metrics,
    load_experiment_summary,
    load_run_result,
)
from mopforge.analysis.normalize import (
    normalize_benchmark_metrics,
    normalize_experiment_rows,
    normalize_run_result,
)
from mopforge.configs.io import MoPForgeConfig
from mopforge.datasets.fingerprint import fingerprint_file
from mopforge.importers.validation import detect_artifacts


IMPORT_KINDS = {"auto", "run", "experiment", "benchmark", "analysis"}


@dataclass(slots=True)
class ResultImportConfig:
    name: str
    source_path: str
    import_kind: str = "auto"
    output_root: str = "imports"
    copy_files: bool = True
    dataset_ref: str | None = None
    model_ref: str | None = None
    manifest_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _non_empty(self.name, "name")
        self.source_path = _non_empty(self.source_path, "source_path")
        if self.import_kind not in IMPORT_KINDS:
            raise ValueError(f"import_kind must be one of: {', '.join(sorted(IMPORT_KINDS))}.")
        self.output_root = _non_empty(self.output_root, "output_root")
        if type(self.copy_files) is not bool:
            raise ValueError("copy_files must be a boolean.")
        self.dataset_ref = _optional(self.dataset_ref, "dataset_ref")
        self.model_ref = _optional(self.model_ref, "model_ref")
        self.manifest_ref = _optional(self.manifest_ref, "manifest_ref")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "import_kind": self.import_kind,
            "output_root": self.output_root,
            "copy_files": self.copy_files,
            "dataset_ref": self.dataset_ref,
            "model_ref": self.model_ref,
            "manifest_ref": self.manifest_ref,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultImportConfig":
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        return MoPForgeConfig(kind="import", payload=self.to_dict()).save(path)


@dataclass(slots=True)
class ResultImportRecord:
    import_id: str
    name: str
    status: str
    created_at: str
    source_path: str
    normalized_results_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_id": self.import_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "source_path": self.source_path,
            "normalized_results_path": self.normalized_results_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultImportRecord":
        return cls(
            import_id=str(data["import_id"]),
            name=str(data["name"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            source_path=str(data["source_path"]),
            normalized_results_path=data.get("normalized_results_path"),
            metadata=dict(data.get("metadata", {})),
        )


class ResultImportRegistry:
    def __init__(self, root: str | Path = "imports") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load_record(self, import_id: str) -> ResultImportRecord:
        path = self.import_dir(import_id) / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"Import record does not exist: {import_id}")
        return ResultImportRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_imports(self) -> list[ResultImportRecord]:
        records = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "record.json"
            if record_path.exists():
                records.append(ResultImportRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8"))))
        return sorted(records, key=lambda record: (record.created_at, record.import_id))

    def import_dir(self, import_id: str) -> Path:
        return self.root / import_id


def import_results(config: ResultImportConfig) -> ResultImportRecord:
    """Import local result artifacts and write normalized rows."""

    config = ResultImportConfig.from_dict(config.to_dict())
    import_id = _import_id(config.name)
    registry = ResultImportRegistry(config.output_root)
    directory = registry.import_dir(import_id)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = detect_artifacts(config.source_path)
    imported_files = []
    for artifact in artifacts:
        source = Path(artifact["path"])
        fingerprint = fingerprint_file(source)
        target = None
        if config.copy_files:
            target = directory / "files" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target = directory / "files" / f"{len(imported_files)}-{source.name}"
            shutil.copy2(source, target)
        imported_files.append(
            {
                **artifact,
                "fingerprint": fingerprint.to_dict(),
                "copied_path": str(target) if target else None,
            }
        )
    rows = _normalize_imported_artifacts(imported_files, config)
    _write_json(directory / "import.json", config.to_dict())
    _write_json(directory / "imported_files.json", imported_files)
    _write_json(directory / "detected_artifacts.json", artifacts)
    normalized_path = _write_json(directory / "normalized_results.json", rows)
    record = ResultImportRecord(
        import_id=import_id,
        name=config.name,
        status="completed",
        created_at=_now(),
        source_path=config.source_path,
        normalized_results_path=str(normalized_path),
        metadata={
            "artifact_count": len(artifacts),
            "row_count": len(rows),
            "dataset_ref": config.dataset_ref,
            "model_ref": config.model_ref,
            "manifest_ref": config.manifest_ref,
        },
    )
    _write_json(directory / "record.json", record.to_dict())
    return record


def _normalize_imported_artifacts(imported_files: list[dict], config: ResultImportConfig) -> list[dict]:
    rows = []
    for item in imported_files:
        path = item.get("copied_path") or item["path"]
        name = item["name"]
        try:
            if name in {"trainer_result.json", "finetune_result.json", "continued_pretrain_result.json"}:
                rows.append(normalize_run_result(load_run_result(path), source_path=path))
            elif name == "summary.json" or name == "summary.csv":
                rows.extend(normalize_experiment_rows(load_experiment_summary(path), source_id=config.name))
            elif name == "metrics.json":
                metrics = load_benchmark_metrics(path)
                if "benchmark_type" in metrics or "benchmark_id" in metrics:
                    rows.extend(normalize_benchmark_metrics(metrics, source_id=metrics.get("benchmark_id") or config.name))
                else:
                    rows.append(normalize_run_result(metrics, source_path=path))
        except Exception as exc:
            rows.append({"source_type": "import_error", "source_id": config.name, "result_path": path, "error": str(exc), "metadata": item})
    for row in rows:
        row.setdefault("metadata", {})
        if isinstance(row["metadata"], dict):
            row["metadata"].update(
                {
                    "import_name": config.name,
                    "dataset_ref": config.dataset_ref,
                    "model_ref": config.model_ref,
                    "manifest_ref": config.manifest_ref,
                }
            )
    return rows


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _import_id(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe or 'import'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty(value, field_name)


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
