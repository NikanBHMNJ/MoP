"""Experiment config schema and deterministic matrix expansion."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mopforge.configs.io import MoPForgeConfig


RUNNABLE_CONFIG_KINDS = {"sft", "pretrain", "trainer"}
MAX_LOCAL_MATRIX_RUNS = 128


@dataclass(slots=True)
class ExperimentConfig:
    """Local CPU experiment config for matrix or explicit-list runs."""

    name: str
    kind: str = "matrix"
    description: str = ""
    base_config: MoPForgeConfig | dict[str, Any] | None = None
    matrix: dict[str, list[Any]] = field(default_factory=dict)
    runs: list[MoPForgeConfig | dict[str, Any]] = field(default_factory=list)
    max_runs: int | None = None
    seed: int = 123
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _require_non_empty(self.name, "name")
        if self.kind not in {"matrix", "list"}:
            raise ValueError("kind must be matrix or list.")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string.")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        if self.max_runs is not None and (
            type(self.max_runs) is not int or self.max_runs <= 0
        ):
            raise ValueError("max_runs must be a positive integer or None.")
        if not isinstance(self.matrix, dict):
            raise ValueError("matrix must be a dictionary.")
        normalized_matrix: dict[str, list[Any]] = {}
        for key, values in self.matrix.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("matrix keys must be non-empty strings.")
            if not isinstance(values, list):
                raise ValueError(f"matrix value for {key!r} must be a list.")
            normalized_matrix[key.strip()] = list(values)
        self.matrix = normalized_matrix
        self.tags = _normalize_strings(self.tags, "tags")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        if self.base_config is not None:
            self.base_config = _coerce_envelope(self.base_config)
            _require_runnable_kind(self.base_config)
        self.runs = [_coerce_envelope(run) for run in self.runs]
        for run in self.runs:
            _require_runnable_kind(run)
        if self.kind == "matrix":
            if self.base_config is None:
                raise ValueError("matrix experiments require base_config.")
            if not self.matrix:
                raise ValueError("matrix experiments require at least one matrix field.")
        if self.kind == "list" and not self.runs:
            raise ValueError("list experiments require at least one run envelope.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable experiment dictionary."""

        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "base_config": (
                self.base_config.to_dict()
                if isinstance(self.base_config, MoPForgeConfig)
                else None
            ),
            "matrix": {key: list(values) for key, values in self.matrix.items()},
            "runs": [
                run.to_dict() if isinstance(run, MoPForgeConfig) else dict(run)
                for run in self.runs
            ],
            "max_runs": self.max_runs,
            "seed": self.seed,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create an experiment config from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("ExperimentConfig.from_dict expects a dictionary.")
        return cls(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "matrix")),
            description=str(data.get("description", "")),
            base_config=data.get("base_config"),
            matrix=dict(data.get("matrix", {}) or {}),
            runs=list(data.get("runs", []) or []),
            max_runs=data.get("max_runs"),
            seed=int(data.get("seed", 123)),
            tags=list(data.get("tags", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def save(self, path: str | Path) -> Path:
        """Save this experiment config as an experiment envelope."""

        envelope = MoPForgeConfig(kind="experiment", payload=self.to_dict())
        return envelope.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load an experiment config from an experiment envelope file."""

        envelope = MoPForgeConfig.load(path)
        if envelope.kind != "experiment":
            raise ValueError(f"Expected kind='experiment', got {envelope.kind!r}.")
        return cls.from_dict(envelope.payload)


def expand_experiment_matrix(config: ExperimentConfig) -> list[MoPForgeConfig]:
    """Expand a matrix/list experiment into runnable config envelopes."""

    config = ExperimentConfig.from_dict(config.to_dict())
    if config.kind == "list":
        runs = [_with_experiment_metadata(run, config, index, {}) for index, run in enumerate(config.runs)]
        return runs[: config.max_runs] if config.max_runs is not None else runs

    assert config.base_config is not None
    keys = list(config.matrix)
    value_lists = [config.matrix[key] for key in keys]
    total_runs = 1
    for values in value_lists:
        total_runs *= len(values)
    effective_runs = (
        min(total_runs, config.max_runs)
        if config.max_runs is not None
        else total_runs
    )
    if effective_runs > MAX_LOCAL_MATRIX_RUNS:
        raise ValueError(
            f"matrix expansion would create {total_runs} runs and execute "
            f"{effective_runs}; set max_runs to limit local CPU execution."
        )

    expanded: list[MoPForgeConfig] = []
    for index, values in enumerate(itertools.product(*value_lists)):
        if config.max_runs is not None and index >= config.max_runs:
            break
        matrix_values = dict(zip(keys, values))
        data = copy.deepcopy(config.base_config.to_dict())
        for path, value in matrix_values.items():
            _set_dotted_path(data, path, value)
        run = MoPForgeConfig.from_dict(data)
        _require_runnable_kind(run)
        expanded.append(_with_experiment_metadata(run, config, index, matrix_values))
    return expanded


def _set_dotted_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(f"Invalid matrix dotted path: {path!r}.")
    if parts[0] not in {"payload", "metadata"}:
        raise ValueError(
            f"Invalid matrix dotted path {path!r}; top-level field must be "
            "payload or metadata."
        )
    target = data
    for part in parts[:-1]:
        existing = target.get(part)
        if existing is None:
            existing = {}
            target[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(
                f"Invalid matrix dotted path {path!r}; {part!r} is not a dictionary."
            )
        target = existing
    target[parts[-1]] = value


def _with_experiment_metadata(
    run: MoPForgeConfig,
    config: ExperimentConfig,
    index: int,
    matrix_values: dict[str, Any],
) -> MoPForgeConfig:
    data = copy.deepcopy(run.to_dict())
    metadata = dict(data.get("metadata", {}) or {})
    metadata["experiment"] = {
        "name": config.name,
        "matrix_index": index,
        "matrix_values": dict(matrix_values),
        "tags": list(config.tags),
    }
    data["metadata"] = metadata
    return MoPForgeConfig.from_dict(data)


def _coerce_envelope(value: MoPForgeConfig | dict[str, Any]) -> MoPForgeConfig:
    if isinstance(value, MoPForgeConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("run/base_config entries must be config envelope dictionaries.")
    return MoPForgeConfig.from_dict(value)


def _require_runnable_kind(config: MoPForgeConfig) -> None:
    if config.kind not in RUNNABLE_CONFIG_KINDS:
        valid = ", ".join(sorted(RUNNABLE_CONFIG_KINDS))
        raise ValueError(f"Experiment child config kind must be one of: {valid}.")


def _normalize_strings(values: list[str], field_name: str) -> list[str]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a list of strings.")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    seen = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
