"""Research run manifest schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mopforge.configs.io import MoPForgeConfig
from mopforge.manifests.resources import ResourceSpec


RUN_KINDS = {"train", "sft", "pretrain", "benchmark", "experiment", "analysis"}
MANIFEST_ACTIONS = {"create"}


@dataclass(slots=True)
class ResearchRunManifest:
    manifest_id: str
    name: str
    created_at: str
    run_kind: str = "train"
    config_ref: str | None = None
    config_payload: dict[str, Any] = field(default_factory=dict)
    model_ref: str | None = None
    dataset_ref: str | None = None
    benchmark_refs: list[str] = field(default_factory=list)
    resource_spec: ResourceSpec = field(default_factory=ResourceSpec)
    command: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    status: str = "planned"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("manifest_id", "name", "created_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.run_kind not in RUN_KINDS:
            raise ValueError(f"run_kind must be one of: {', '.join(sorted(RUN_KINDS))}.")
        if not isinstance(self.config_payload, dict):
            raise ValueError("config_payload must be a dictionary.")
        if not isinstance(self.resource_spec, ResourceSpec):
            self.resource_spec = ResourceSpec.from_dict(self.resource_spec)
        if not isinstance(self.command, list) or not all(isinstance(item, str) for item in self.command):
            raise ValueError("command must be a list of strings.")
        if self.status not in {"planned", "exported", "imported", "cancelled"}:
            raise ValueError("status is not supported.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "name": self.name,
            "created_at": self.created_at,
            "run_kind": self.run_kind,
            "config_ref": self.config_ref,
            "config_payload": dict(self.config_payload),
            "model_ref": self.model_ref,
            "dataset_ref": self.dataset_ref,
            "benchmark_refs": list(self.benchmark_refs),
            "resource_spec": self.resource_spec.to_dict(),
            "command": list(self.command),
            "environment": dict(self.environment),
            "expected_outputs": list(self.expected_outputs),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchRunManifest":
        return cls(
            manifest_id=str(data["manifest_id"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            run_kind=str(data.get("run_kind", "train")),
            config_ref=data.get("config_ref"),
            config_payload=dict(data.get("config_payload", {})),
            model_ref=data.get("model_ref"),
            dataset_ref=data.get("dataset_ref"),
            benchmark_refs=list(data.get("benchmark_refs", [])),
            resource_spec=ResourceSpec.from_dict(data.get("resource_spec", {})),
            command=list(data.get("command", [])),
            environment={str(k): str(v) for k, v in data.get("environment", {}).items()},
            expected_outputs=list(data.get("expected_outputs", [])),
            status=str(data.get("status", "planned")),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ResearchRunManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class ManifestConfig:
    action: str = "create"
    name: str = "run_manifest"
    config_ref: str | None = None
    config_payload: dict[str, Any] = field(default_factory=dict)
    resource_spec: dict[str, Any] = field(default_factory=dict)
    output_root: str = "manifests"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in MANIFEST_ACTIONS:
            raise ValueError("action must be create.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty.")
        if self.config_ref is not None and (not isinstance(self.config_ref, str) or not self.config_ref.strip()):
            raise ValueError("config_ref must be non-empty or None.")
        if not isinstance(self.config_payload, dict):
            raise ValueError("config_payload must be a dictionary.")
        ResourceSpec.from_dict(self.resource_spec or {})
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be non-empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "name": self.name,
            "config_ref": self.config_ref,
            "config_payload": dict(self.config_payload),
            "resource_spec": dict(self.resource_spec),
            "output_root": self.output_root,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestConfig":
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        return MoPForgeConfig(kind="manifest", payload=self.to_dict()).save(path)
