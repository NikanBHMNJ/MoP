"""Model manifest schemas for local model registry entries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mopforge.configs.io import MoPForgeConfig
from mopforge.models.architectures import ModelArchitectureConfig


MODEL_ACTIONS = {"register", "snapshot"}
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(slots=True)
class ModelManifest:
    model_id: str
    version_id: str
    name: str
    architecture: ModelArchitectureConfig
    created_at: str
    parameter_summary: dict[str, Any] = field(default_factory=dict)
    checkpoint_ref: str | None = None
    dataset_ref: str | None = None
    tokenizer_ref: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_id = slugify_model_id(self.model_id)
        self.version_id = _non_empty(self.version_id, "version_id")
        self.name = _non_empty(self.name, "name")
        if not isinstance(self.architecture, ModelArchitectureConfig):
            self.architecture = ModelArchitectureConfig.from_dict(self.architecture)
        self.created_at = _non_empty(self.created_at, "created_at")
        if not isinstance(self.parameter_summary, dict):
            raise ValueError("parameter_summary must be a dictionary.")
        json.dumps(self.parameter_summary, sort_keys=True)
        self.checkpoint_ref = _optional(self.checkpoint_ref, "checkpoint_ref")
        self.dataset_ref = _optional(self.dataset_ref, "dataset_ref")
        self.tokenizer_ref = _optional(self.tokenizer_ref, "tokenizer_ref")
        self.tags = _strings(self.tags, "tags")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version_id": self.version_id,
            "name": self.name,
            "architecture": self.architecture.to_dict(),
            "created_at": self.created_at,
            "parameter_summary": dict(self.parameter_summary),
            "checkpoint_ref": self.checkpoint_ref,
            "dataset_ref": self.dataset_ref,
            "tokenizer_ref": self.tokenizer_ref,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelManifest":
        return cls(
            model_id=str(data["model_id"]),
            version_id=str(data["version_id"]),
            name=str(data["name"]),
            architecture=ModelArchitectureConfig.from_dict(data["architecture"]),
            created_at=str(data["created_at"]),
            parameter_summary=dict(data.get("parameter_summary", {})),
            checkpoint_ref=data.get("checkpoint_ref"),
            dataset_ref=data.get("dataset_ref"),
            tokenizer_ref=data.get("tokenizer_ref"),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class ModelConfig:
    action: str = "register"
    name: str = ""
    model_id: str | None = None
    architecture: dict[str, Any] = field(default_factory=dict)
    checkpoint_ref: str | None = None
    output_root: str = "models"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in MODEL_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(sorted(MODEL_ACTIONS))}.")
        self.name = _optional(self.name, "name") or ""
        if self.model_id is not None:
            self.model_id = slugify_model_id(self.model_id)
        if not isinstance(self.architecture, dict):
            raise ValueError("architecture must be a dictionary.")
        self.checkpoint_ref = _optional(self.checkpoint_ref, "checkpoint_ref")
        self.output_root = _non_empty(self.output_root, "output_root")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "name": self.name,
            "model_id": self.model_id,
            "architecture": dict(self.architecture),
            "checkpoint_ref": self.checkpoint_ref,
            "output_root": self.output_root,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        return MoPForgeConfig(kind="model", payload=self.to_dict()).save(path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        envelope = MoPForgeConfig.load(path)
        if envelope.kind != "model":
            raise ValueError(f"Expected kind='model', got {envelope.kind!r}.")
        return cls.from_dict(envelope.payload)


def slugify_model_id(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model_id must be a non-empty string.")
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    if not _SLUG_RE.match(slug):
        raise ValueError("model_id must be slug-like.")
    return slug


def _strings(values: list[str], field_name: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    return [value.strip() for value in values]


def _optional(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None.")
    return value.strip() or None


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
