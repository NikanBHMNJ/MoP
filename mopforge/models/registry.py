"""Local model registry and versioning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mopforge.models.architectures import (
    ModelArchitectureConfig,
    parameter_summary_for_architecture,
)
from mopforge.models.manifest import ModelManifest, slugify_model_id


@dataclass(slots=True)
class ModelRecord:
    model_id: str
    name: str
    model_type: str
    created_at: str
    updated_at: str
    latest_version_id: str | None = None
    versions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("model_id", "name", "model_type", "created_at", "updated_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.versions, list):
            raise ValueError("versions must be a list.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_version_id": self.latest_version_id,
            "versions": list(self.versions),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        return cls(
            model_id=str(data["model_id"]),
            name=str(data["name"]),
            model_type=str(data["model_type"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            latest_version_id=data.get("latest_version_id"),
            versions=list(data.get("versions", [])),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )


class ModelRegistry:
    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        if not self.registry_path.exists():
            _write_json(self.registry_path, {"models": []})

    def register_model(
        self,
        architecture,
        model_id=None,
        tags=None,
        metadata=None,
    ) -> ModelManifest:
        architecture = (
            architecture
            if isinstance(architecture, ModelArchitectureConfig)
            else ModelArchitectureConfig.from_dict(architecture)
        )
        model_id = slugify_model_id(model_id or architecture.name)
        record_path = self.model_dir(model_id) / "model.json"
        now = _now()
        if record_path.exists():
            record = self.load_model_record(model_id)
        else:
            record = ModelRecord(
                model_id=model_id,
                name=architecture.name,
                model_type=architecture.model_type,
                created_at=now,
                updated_at=now,
                tags=list(tags or []),
                metadata=dict(metadata or {}),
            )
        return self._create_manifest(
            record=record,
            architecture=architecture,
            checkpoint_ref=None,
            metadata=dict(metadata or {}),
        )

    def snapshot_model(
        self,
        model_id,
        architecture=None,
        checkpoint_ref=None,
        metadata=None,
    ) -> ModelManifest:
        record = self.load_model_record(model_id)
        if architecture is None:
            architecture = self.load_manifest(model_id).architecture
        return self._create_manifest(
            record=record,
            architecture=architecture,
            checkpoint_ref=checkpoint_ref,
            metadata=dict(metadata or {}),
        )

    def load_model_record(self, model_id) -> ModelRecord:
        path = self.model_dir(model_id) / "model.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Model record does not exist: {model_id}. "
                "Register it with `mopforge model register <config>` or run `mopforge model list`."
            )
        return ModelRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_manifest(self, model_id, version_id=None) -> ModelManifest:
        record = self.load_model_record(model_id)
        version_id = version_id or record.latest_version_id
        if not version_id:
            raise FileNotFoundError(
                f"Model has no versions: {model_id}. "
                "Create one with `mopforge model register` or `mopforge model snapshot`."
            )
        path = self.version_dir(model_id, version_id) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Model manifest does not exist: {model_id}@{version_id}. "
                "Check the model ref or run `mopforge model versions <model_id>`."
            )
        manifest = ModelManifest.load(path)
        manifest.metadata.setdefault("manifest_path", str(path))
        manifest.metadata.setdefault("version_dir", str(path.parent))
        return manifest

    def list_models(self) -> list[ModelRecord]:
        records = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "model.json"
            if record_path.exists():
                records.append(ModelRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8"))))
        return sorted(records, key=lambda record: (record.created_at, record.model_id))

    def list_versions(self, model_id) -> list[ModelManifest]:
        record = self.load_model_record(model_id)
        return [self.load_manifest(model_id, version) for version in record.versions]

    def resolve_model_ref(self, model_ref: str) -> ModelManifest:
        candidate = Path(model_ref)
        if candidate.exists() and candidate.is_file():
            manifest = ModelManifest.load(candidate)
            manifest.metadata.setdefault("manifest_path", str(candidate))
            manifest.metadata.setdefault("version_dir", str(candidate.parent))
            return manifest
        if "@" in model_ref:
            model_id, version_id = model_ref.split("@", 1)
            return self.load_manifest(model_id, version_id)
        return self.load_manifest(model_ref)

    def model_dir(self, model_id) -> Path:
        return self.root / str(model_id)

    def version_dir(self, model_id, version_id) -> Path:
        return self.model_dir(model_id) / "versions" / str(version_id)

    def _create_manifest(
        self,
        *,
        record: ModelRecord,
        architecture: ModelArchitectureConfig,
        checkpoint_ref: str | None,
        metadata: dict[str, Any],
    ) -> ModelManifest:
        summary = parameter_summary_for_architecture(architecture)
        version_id = _version_id(architecture)
        index = 1
        base = version_id
        while self.version_dir(record.model_id, version_id).exists():
            index += 1
            version_id = f"{base}-{index}"
        version_dir = self.version_dir(record.model_id, version_id)
        manifest = ModelManifest(
            model_id=record.model_id,
            version_id=version_id,
            name=architecture.name,
            architecture=architecture,
            created_at=_now(),
            parameter_summary=summary,
            checkpoint_ref=checkpoint_ref,
            dataset_ref=architecture.dataset_ref,
            tokenizer_ref=architecture.tokenizer_ref,
            tags=list(record.tags),
            metadata={
                **metadata,
                "manifest_path": str(version_dir / "manifest.json"),
                "version_dir": str(version_dir),
            },
        )
        manifest.save(version_dir / "manifest.json")
        architecture.save(version_dir / "architecture.json")
        _write_json(version_dir / "parameter_summary.json", summary)
        if version_id not in record.versions:
            record.versions.append(version_id)
        record.latest_version_id = version_id
        record.updated_at = _now()
        record.model_type = architecture.model_type
        _write_json(self.model_dir(record.model_id) / "model.json", record.to_dict())
        _write_json(self.registry_path, {"models": [item.to_dict() for item in self.list_models()]})
        return manifest


def _version_id(architecture: ModelArchitectureConfig) -> str:
    digest = hashlib.sha256(json.dumps(architecture.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    slug = slugify_model_id(architecture.name).replace("_", "-")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{slug}-{digest[:8]}"


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
