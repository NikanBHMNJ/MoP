"""Resource specifications for future run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ACCELERATORS = {"none", "cpu", "cuda", "a100_80gb", "h100_80gb", "b300"}
PRECISIONS = {"fp32", "fp16", "bf16", "fp8"}


@dataclass(slots=True)
class ResourceSpec:
    accelerator: str = "none"
    num_gpus: int = 0
    gpu_memory_gb: int | None = None
    precision: str = "fp32"
    nodes: int = 1
    cpus: int | None = None
    memory_gb: int | None = None
    max_runtime_hours: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accelerator not in ACCELERATORS:
            raise ValueError(f"accelerator must be one of: {', '.join(sorted(ACCELERATORS))}.")
        if type(self.num_gpus) is not int or self.num_gpus < 0:
            raise ValueError("num_gpus must be a non-negative integer.")
        if self.accelerator not in {"none", "cpu"} and self.num_gpus <= 0:
            raise ValueError("GPU accelerator plans require num_gpus > 0.")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of: {', '.join(sorted(PRECISIONS))}.")
        if type(self.nodes) is not int or self.nodes <= 0:
            raise ValueError("nodes must be a positive integer.")
        for field_name in ("gpu_memory_gb", "cpus", "memory_gb"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field_name} must be a positive integer or None.")
        if self.max_runtime_hours is not None and self.max_runtime_hours <= 0:
            raise ValueError("max_runtime_hours must be positive or None.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceSpec":
        return cls(**dict(data))
