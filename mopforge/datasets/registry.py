"""File-backed local dataset registry."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mopforge.datasets.fingerprint import (
    combined_fingerprint,
    fingerprint_files,
)
from mopforge.datasets.manifest import (
    DatasetManifest,
    slugify_dataset_id,
)
from mopforge.datasets.stats import compute_dataset_stats


@dataclass(slots=True)
class DatasetRecord:
    """Mutable record for a registered local dataset."""

    dataset_id: str
    name: str
    kind: str
    created_at: str
    updated_at: str
    latest_version_id: str | None = None
    versions: list[str] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if this record is malformed."""

        for field_name in ("dataset_id", "name", "kind", "created_at", "updated_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.versions, list) or not all(
            isinstance(version, str) for version in self.versions
        ):
            raise ValueError("versions must be a list of strings.")
        if not isinstance(self.tags, list) or not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("tags must be a list of strings.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "kind": self.kind,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_version_id": self.latest_version_id,
            "versions": list(self.versions),
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecord":
        """Create a dataset record from a dictionary."""

        return cls(
            dataset_id=str(data["dataset_id"]),
            name=str(data["name"]),
            kind=str(data["kind"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            latest_version_id=data.get("latest_version_id"),
            versions=list(data.get("versions", [])),
            description=str(data.get("description", "")),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )


class DatasetRegistry:
    """Local dataset registry rooted at ``datasets/``."""

    def __init__(self, root: str | Path = "datasets") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        if not self.registry_path.exists():
            _write_json(self.registry_path, {"datasets": []})

    def register_dataset(
        self,
        name,
        kind,
        source_paths,
        dataset_id=None,
        description="",
        tags=None,
        metadata=None,
        copy_files=False,
    ) -> DatasetManifest:
        """Register a dataset and create its first version manifest."""

        if not name or not str(name).strip():
            raise ValueError("name must be a non-empty string.")
        source_paths = _source_paths(source_paths)
        dataset_id = dataset_id or slugify_dataset_id(str(name))
        tags = list(tags or [])
        metadata = dict(metadata or {})
        now = _now()
        record_path = self.dataset_dir(dataset_id) / "dataset.json"
        if record_path.exists():
            record = self.load_dataset_record(dataset_id)
        else:
            record = DatasetRecord(
                dataset_id=dataset_id,
                name=str(name).strip(),
                kind=str(kind),
                created_at=now,
                updated_at=now,
                description=str(description or ""),
                tags=tags,
                metadata={},
            )
        record.metadata.update(
            {
                "source_paths": [str(Path(path).expanduser().resolve()) for path in source_paths],
                **metadata,
            }
        )
        manifest = self._create_manifest(
            record=record,
            source_paths=source_paths,
            copy_files=copy_files,
            metadata=metadata,
        )
        self._record_version(record, manifest.version_id)
        return manifest

    def snapshot_dataset(
        self,
        dataset_id,
        source_paths=None,
        copy_files=False,
        metadata=None,
    ) -> DatasetManifest:
        """Create a new version manifest for an existing dataset."""

        record = self.load_dataset_record(str(dataset_id))
        if source_paths is None:
            source_paths = record.metadata.get("source_paths")
        source_paths = _source_paths(source_paths)
        record.metadata["source_paths"] = [
            str(Path(path).expanduser().resolve()) for path in source_paths
        ]
        manifest = self._create_manifest(
            record=record,
            source_paths=source_paths,
            copy_files=copy_files,
            metadata=dict(metadata or {}),
        )
        self._record_version(record, manifest.version_id)
        return manifest

    def load_dataset_record(self, dataset_id) -> DatasetRecord:
        """Load a dataset record by ID."""

        path = self.dataset_dir(str(dataset_id)) / "dataset.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset record does not exist: {dataset_id}. "
                "Register it with `mopforge dataset register ...` or run `mopforge dataset list`."
            )
        return DatasetRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_manifest(self, dataset_id, version_id=None) -> DatasetManifest:
        """Load a manifest by dataset/version, defaulting to latest."""

        record = self.load_dataset_record(str(dataset_id))
        version_id = version_id or record.latest_version_id
        if not version_id:
            raise FileNotFoundError(
                f"Dataset has no versions: {dataset_id}. "
                "Create one with `mopforge dataset register` or `mopforge dataset snapshot`."
            )
        path = self.version_dir(record.dataset_id, version_id) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset manifest does not exist: {dataset_id}@{version_id}. "
                "Check the dataset ref or run `mopforge dataset versions <dataset_id>`."
            )
        manifest = DatasetManifest.load(path)
        manifest.metadata.setdefault("version_dir", str(path.parent))
        manifest.metadata.setdefault("manifest_path", str(path))
        return manifest

    def list_datasets(self) -> list[DatasetRecord]:
        """List local dataset records."""

        records: list[DatasetRecord] = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "dataset.json"
            if record_path.exists():
                records.append(
                    DatasetRecord.from_dict(
                        json.loads(record_path.read_text(encoding="utf-8"))
                    )
                )
        return sorted(records, key=lambda record: (record.created_at, record.dataset_id))

    def list_versions(self, dataset_id) -> list[DatasetManifest]:
        """List manifests for one dataset in record order."""

        record = self.load_dataset_record(str(dataset_id))
        return [self.load_manifest(record.dataset_id, version) for version in record.versions]

    def resolve_dataset_ref(self, dataset_ref: str) -> DatasetManifest:
        """Resolve ``dataset_id``, ``dataset_id@version_id``, or manifest path."""

        if not isinstance(dataset_ref, str) or not dataset_ref.strip():
            raise ValueError("dataset_ref must be a non-empty string.")
        candidate = Path(dataset_ref)
        if candidate.exists() and candidate.is_file():
            manifest = DatasetManifest.load(candidate)
            manifest.metadata.setdefault("version_dir", str(candidate.parent))
            manifest.metadata.setdefault("manifest_path", str(candidate))
            return manifest
        if "@" in dataset_ref:
            dataset_id, version_id = dataset_ref.split("@", 1)
            return self.load_manifest(dataset_id, version_id)
        return self.load_manifest(dataset_ref)

    def dataset_dir(self, dataset_id) -> Path:
        """Return a dataset directory path."""

        return self.root / str(dataset_id)

    def version_dir(self, dataset_id, version_id) -> Path:
        """Return a version directory path."""

        return self.dataset_dir(dataset_id) / "versions" / str(version_id)

    def _create_manifest(
        self,
        *,
        record: DatasetRecord,
        source_paths: list[Path],
        copy_files: bool,
        metadata: dict[str, Any],
    ) -> DatasetManifest:
        fingerprints = fingerprint_files(source_paths)
        combined = combined_fingerprint(fingerprints)
        version_id = self._make_version_id(record.dataset_id, record.name, combined)
        version_dir = self.version_dir(record.dataset_id, version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        copied_paths: list[str] = []
        if copy_files:
            files_dir = version_dir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            for index, source_path in enumerate(source_paths):
                destination = files_dir / _copy_name(source_path, index)
                shutil.copy2(source_path, destination)
                copied_paths.append(str(destination.resolve()))
        manifest_metadata = {
            **metadata,
            "file_storage": "copied" if copy_files else "referenced",
            "original_source_paths": [fingerprint.path for fingerprint in fingerprints],
            "copied_source_paths": copied_paths,
            "version_dir": str(version_dir),
            "manifest_path": str(version_dir / "manifest.json"),
        }
        stats = compute_dataset_stats(source_paths[0], record.kind)
        if len(source_paths) > 1:
            stats.metadata["additional_source_paths"] = [
                str(path.resolve()) for path in source_paths[1:]
            ]
        manifest = DatasetManifest(
            dataset_id=record.dataset_id,
            version_id=version_id,
            name=record.name,
            kind=record.kind,
            created_at=_now(),
            source_paths=[fingerprint.path for fingerprint in fingerprints],
            fingerprints=fingerprints,
            combined_sha256=combined,
            stats=stats,
            description=record.description,
            tags=list(record.tags),
            metadata=manifest_metadata,
        )
        manifest.save(version_dir / "manifest.json")
        _write_json(version_dir / "stats.json", stats.to_dict())
        return manifest

    def _record_version(self, record: DatasetRecord, version_id: str) -> None:
        if version_id not in record.versions:
            record.versions.append(version_id)
        record.latest_version_id = version_id
        record.updated_at = _now()
        self._save_record(record)
        self._write_global_registry()

    def _save_record(self, record: DatasetRecord) -> None:
        record.validate()
        path = self.dataset_dir(record.dataset_id) / "dataset.json"
        _write_json(path, record.to_dict())

    def _write_global_registry(self) -> None:
        _write_json(
            self.registry_path,
            {"datasets": [record.to_dict() for record in self.list_datasets()]},
        )

    def _make_version_id(self, dataset_id: str, name: str, combined_sha256: str) -> str:
        slug = slugify_dataset_id(name).replace("_", "-")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"{timestamp}-{slug}-{combined_sha256[:8]}"
        candidate = base
        index = 1
        while self.version_dir(dataset_id, candidate).exists():
            index += 1
            candidate = f"{base}-{index}"
        return candidate


def _source_paths(source_paths) -> list[Path]:
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("source_paths must be a non-empty list.")
    paths = []
    for path in source_paths:
        candidate = Path(path).expanduser()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Dataset source file does not exist: {path}")
        paths.append(candidate)
    return paths


def _copy_name(source_path: Path, index: int) -> str:
    name = source_path.name or f"source-{index}.jsonl"
    if index == 0:
        return name
    return f"{index}-{name}"


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
