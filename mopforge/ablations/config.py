"""Ablation config schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mopforge.configs.io import MoPForgeConfig


@dataclass(slots=True)
class AblationVariant:
    name: str
    description: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("variant name must be non-empty.")
        if not isinstance(self.overrides, dict):
            raise ValueError("overrides must be a dictionary.")
        if not isinstance(self.tags, list):
            raise ValueError("tags must be a list.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationVariant":
        return cls(**dict(data))


@dataclass(slots=True)
class AblationConfig:
    name: str
    description: str = ""
    base_config: dict[str, Any] = field(default_factory=dict)
    variants: list[AblationVariant] = field(default_factory=list)
    benchmark_configs: list[dict[str, Any]] = field(default_factory=list)
    rank_by: str | None = "final_eval_loss"
    rank_mode: str = "min"
    output_root: str = "ablations"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty.")
        if not isinstance(self.base_config, dict):
            raise ValueError("base_config must be a dictionary.")
        self.variants = [
            item if isinstance(item, AblationVariant) else AblationVariant.from_dict(item)
            for item in self.variants
        ]
        if not self.variants:
            raise ValueError("variants must be non-empty.")
        if self.rank_mode not in {"min", "max"}:
            raise ValueError("rank_mode must be min or max.")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be non-empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variants"] = [variant.to_dict() for variant in self.variants]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationConfig":
        return cls(**dict(data))

    def save(self, path) -> object:
        return MoPForgeConfig(kind="ablation", payload=self.to_dict()).save(path)
