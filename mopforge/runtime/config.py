"""Runtime configuration for device and precision selection."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_PRECISIONS = {"fp32", "fp16", "bf16", "fp8", "auto"}
_CUDA_PATTERN = re.compile(r"^cuda(?::\d+)?$")


@dataclass(slots=True)
class RuntimeConfig:
    """Device/precision runtime request.

    The config is intentionally import-safe on CPU-only systems. Availability
    checks happen in ``resolve_device`` rather than during dataclass creation.
    """

    device: str = "cpu"
    precision: str = "fp32"
    enable_amp: bool = False
    allow_tf32: bool = False
    deterministic: bool = False
    compile_model: bool = False
    require_device_available: bool = True
    log_runtime_warnings: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")
        self.device = self.device.strip().lower()
        if not _valid_device(self.device):
            raise ValueError("device must be cpu, auto, cuda, cuda:<index>, or mps.")
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be a non-empty string.")
        self.precision = self.precision.strip().lower()
        if self.precision not in VALID_PRECISIONS:
            raise ValueError(
                "precision must be one of: auto, bf16, fp16, fp32, fp8."
            )
        for field_name in (
            "enable_amp",
            "allow_tf32",
            "deterministic",
            "compile_model",
            "require_device_available",
            "log_runtime_warnings",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        json.dumps(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        """Create a runtime config from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("RuntimeConfig.from_dict expects a dictionary.")
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        """Save this config as JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        """Load this config from JSON."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def runtime_config_from_kwargs(**kwargs: Any) -> RuntimeConfig:
    """Build ``RuntimeConfig`` from keyword arguments."""

    return RuntimeConfig(**kwargs)


def _valid_device(device: str) -> bool:
    return device in {"cpu", "auto", "mps"} or bool(_CUDA_PATTERN.match(device))
