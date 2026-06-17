"""Baseline specification schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BASELINE_FAMILIES = {"dense", "adapter", "generated", "mop_oracle", "mop_learned_router", "moe"}


@dataclass(slots=True)
class BaselineSpec:
    name: str
    family: str
    model_type: str
    trainable_policy_mode: str = "all"
    use_fast_adapters: bool = False
    use_generated_params: bool = False
    routing_mode: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty.")
        if self.family not in BASELINE_FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(sorted(BASELINE_FAMILIES))}.")
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be non-empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineSpec":
        return cls(**dict(data))


@dataclass(slots=True)
class BaselineConfig:
    action: str = "experiment"
    baselines: list[str] = field(default_factory=lambda: ["dense_full", "adapter_only", "generated_params_only", "mop_module_only"])
    output_root: str = "experiments"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in {"experiment"}:
            raise ValueError("action must be experiment.")
        if not isinstance(self.baselines, list) or not self.baselines:
            raise ValueError("baselines must be a non-empty list.")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be non-empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineConfig":
        return cls(**dict(data))
