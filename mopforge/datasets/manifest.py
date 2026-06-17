"""Dataset manifest and config schemas."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mopforge.configs.io import MoPForgeConfig
from mopforge.datasets.fingerprint import FileFingerprint
from mopforge.datasets.stats import DatasetStats, KNOWN_DATASET_KINDS


DATASET_ACTIONS = {"register", "snapshot", "split"}
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(slots=True)
class DatasetManifest:
    """Immutable-ish local dataset version manifest."""

    dataset_id: str
    version_id: str
    name: str
    kind: str
    created_at: str
    source_paths: list[str]
    fingerprints: list[FileFingerprint]
    combined_sha256: str
    stats: DatasetStats
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_id = _require_slug(self.dataset_id, "dataset_id")
        self.version_id = _require_non_empty(self.version_id, "version_id")
        self.name = _require_non_empty(self.name, "name")
        if self.kind not in KNOWN_DATASET_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(sorted(KNOWN_DATASET_KINDS))}.")
        self.created_at = _require_non_empty(self.created_at, "created_at")
        self.source_paths = _string_list(self.source_paths, "source_paths", require_non_empty=True)
        if not isinstance(self.fingerprints, list) or not self.fingerprints:
            raise ValueError("fingerprints must be a non-empty list.")
        self.fingerprints = [
            item if isinstance(item, FileFingerprint) else FileFingerprint.from_dict(item)
            for item in self.fingerprints
        ]
        self.combined_sha256 = _require_non_empty(self.combined_sha256, "combined_sha256")
        if len(self.combined_sha256) != 64:
            raise ValueError("combined_sha256 must be a 64-character hex string.")
        if not isinstance(self.stats, DatasetStats):
            self.stats = DatasetStats.from_dict(self.stats)
        if self.stats.kind != self.kind:
            raise ValueError("stats kind must match dataset kind.")
        self.description = _optional_string(self.description, "description") or ""
        self.tags = _string_list(self.tags, "tags")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return {
            "dataset_id": self.dataset_id,
            "version_id": self.version_id,
            "name": self.name,
            "kind": self.kind,
            "created_at": self.created_at,
            "source_paths": list(self.source_paths),
            "fingerprints": [fingerprint.to_dict() for fingerprint in self.fingerprints],
            "combined_sha256": self.combined_sha256,
            "stats": self.stats.to_dict(),
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        """Create a manifest from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("DatasetManifest.from_dict expects a dictionary.")
        return cls(
            dataset_id=str(data["dataset_id"]),
            version_id=str(data["version_id"]),
            name=str(data["name"]),
            kind=str(data["kind"]),
            created_at=str(data["created_at"]),
            source_paths=list(data.get("source_paths", [])),
            fingerprints=[
                FileFingerprint.from_dict(item)
                for item in data.get("fingerprints", [])
            ],
            combined_sha256=str(data["combined_sha256"]),
            stats=DatasetStats.from_dict(data["stats"]),
            description=str(data.get("description", "")),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Save this manifest JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        """Load a manifest JSON file."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class DatasetConfig:
    """Config envelope payload for dataset registry actions."""

    action: str = "register"
    name: str = ""
    dataset_id: str | None = None
    kind: str = "lessons"
    source_paths: list[str] = field(default_factory=list)
    dataset_ref: str | None = None
    copy_files: bool = False
    split_train: float = 0.8
    split_eval: float = 0.1
    split_test: float = 0.1
    split_seed: int = 123
    stratify_by: str | None = None
    output_root: str = "datasets"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in DATASET_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(sorted(DATASET_ACTIONS))}.")
        self.name = _optional_string(self.name, "name") or ""
        if self.dataset_id is not None:
            self.dataset_id = _require_slug(self.dataset_id, "dataset_id")
        if self.kind not in KNOWN_DATASET_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(sorted(KNOWN_DATASET_KINDS))}.")
        self.source_paths = _string_list(self.source_paths, "source_paths")
        if self.dataset_ref is not None:
            self.dataset_ref = _require_non_empty(self.dataset_ref, "dataset_ref")
        if type(self.copy_files) is not bool:
            raise ValueError("copy_files must be a boolean.")
        for field_name in ("split_train", "split_eval", "split_test"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative number.")
            setattr(self, field_name, float(value))
        if type(self.split_seed) is not int:
            raise ValueError("split_seed must be an integer.")
        if self.stratify_by is not None:
            self.stratify_by = _require_non_empty(self.stratify_by, "stratify_by")
        self.output_root = _require_non_empty(self.output_root, "output_root")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        """Create a dataset config from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("DatasetConfig.from_dict expects a dictionary.")
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        """Save this config as a dataset envelope."""

        return MoPForgeConfig(kind="dataset", payload=self.to_dict()).save(path)

    @classmethod
    def load(cls, path: str | Path) -> "DatasetConfig":
        """Load a dataset config envelope."""

        envelope = MoPForgeConfig.load(path)
        if envelope.kind != "dataset":
            raise ValueError(f"Expected kind='dataset', got {envelope.kind!r}.")
        return cls.from_dict(envelope.payload)


def slugify_dataset_id(name: str) -> str:
    """Convert a human name into a local dataset ID."""

    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return _require_slug(slug or "dataset", "dataset_id")


def _require_slug(value: str, field_name: str) -> str:
    value = _require_non_empty(value, field_name)
    if not _SLUG_RE.match(value):
        raise ValueError(f"{field_name} must be slug-like.")
    return value


def _string_list(
    values: list[str],
    field_name: str,
    *,
    require_non_empty: bool = False,
) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    if require_non_empty and not values:
        raise ValueError(f"{field_name} must be non-empty.")
    return [_require_non_empty(value, field_name) for value in values]


def _optional_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value.strip()


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
