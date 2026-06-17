"""Analysis config schema for local result reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mopforge.configs.io import MoPForgeConfig


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration for local analysis/report generation."""

    name: str
    description: str = ""
    experiment_ids: list[str] = field(default_factory=list)
    benchmark_ids: list[str] = field(default_factory=list)
    run_paths: list[str] = field(default_factory=list)
    output_root: str = "reports"
    metrics: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    rank_by: str | None = None
    rank_mode: str = "min"
    baseline_filter: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _require_non_empty(self.name, "name")
        self.description = _optional_string(self.description, "description") or ""
        self.experiment_ids = _string_list(self.experiment_ids, "experiment_ids")
        self.benchmark_ids = _string_list(self.benchmark_ids, "benchmark_ids")
        self.run_paths = _string_list(self.run_paths, "run_paths")
        self.metrics = _string_list(self.metrics, "metrics")
        self.group_by = _string_list(self.group_by, "group_by")
        self.output_root = _require_non_empty(self.output_root, "output_root")
        if self.rank_by is not None:
            self.rank_by = _require_non_empty(self.rank_by, "rank_by")
        if self.rank_mode not in {"min", "max"}:
            raise ValueError("rank_mode must be 'min' or 'max'.")
        if not isinstance(self.baseline_filter, dict):
            raise ValueError("baseline_filter must be a dictionary.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        """Create an analysis config from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("AnalysisConfig.from_dict expects a dictionary.")
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        """Save this config as an analysis envelope."""

        return MoPForgeConfig(kind="analysis", payload=self.to_dict()).save(path)

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        """Load an analysis config from an envelope file."""

        envelope = MoPForgeConfig.load(path)
        if envelope.kind != "analysis":
            raise ValueError(f"Expected kind='analysis', got {envelope.kind!r}.")
        return cls.from_dict(envelope.payload)


def _string_list(values: list[str], field_name: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    result: list[str] = []
    for value in values:
        result.append(_require_non_empty(value, field_name))
    return result


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
