"""Config file IO and envelope schema for MoP-Forge."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_CONFIG_KINDS = {
    "trainer",
    "sft",
    "pretrain",
    "experiment",
    "benchmark",
    "analysis",
    "dataset",
    "model",
    "manifest",
    "import",
    "ablation",
    "baseline",
    "stats",
    "paper_report",
    "runtime",
    "gpu_train",
    "queue",
}


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or optional YAML config file."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {input_path}. "
            "Check the path or run `mopforge config write-default <name> <path>`."
        )
    if not input_path.is_file():
        raise ValueError(f"Config path is not a file: {input_path}.")
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON config {input_path}: {exc.msg} "
                f"at line {exc.lineno}, column {exc.colno}."
            ) from exc
    elif suffix in {".yaml", ".yml"}:
        yaml = _require_yaml()
        try:
            data = yaml.safe_load(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid YAML config {input_path}: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported config suffix {suffix!r}. Use .json, .yaml, or .yml."
        )
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a dictionary/object at the top level.")
    return dict(data)


def save_config_file(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as JSON or optional YAML."""

    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        output_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if suffix in {".yaml", ".yml"}:
        yaml = _require_yaml()
        output_path.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
        return
    raise ValueError(
        f"Unsupported config suffix {suffix!r}. Use .json, .yaml, or .yml."
    )


@dataclass(slots=True)
class MoPForgeConfig:
    """Generic config envelope for CLI and file-based runs."""

    kind: str
    version: str = "1"
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate envelope shape."""

        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be a non-empty string.")
        self.kind = self.kind.strip()
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string.")
        self.version = self.version.strip()
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        _ensure_json_serializable(self.payload, "payload")
        _ensure_json_serializable(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as a JSON-serializable dictionary."""

        return {
            "kind": self.kind,
            "version": self.version,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoPForgeConfig":
        """Create an envelope from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("MoPForgeConfig.from_dict expects a dictionary.")
        return cls(
            kind=data.get("kind", ""),
            version=str(data.get("version", "1")),
            payload=dict(data.get("payload", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def save(self, path: str | Path) -> Path:
        """Save this envelope and return the output path."""

        output_path = Path(path)
        save_config_file(self.to_dict(), output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "MoPForgeConfig":
        """Load a config envelope from a file."""

        return cls.from_dict(load_config_file(path))


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML config support requires optional dependency PyYAML. "
            "Install it with: pip install PyYAML"
        ) from exc
    return yaml


def _ensure_json_serializable(value: Any, field_name: str) -> None:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be JSON-serializable.") from exc
