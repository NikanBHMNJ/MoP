"""Curated public API policy for MoP-Forge.

The top-level ``mopforge`` package keeps broad legacy exports for backwards
compatibility. This module is the smaller, documented surface that new users
should prefer when they want stable imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mopforge.configs import MoPForgeConfig, get_default_config, list_default_config_names
from mopforge.gpu import (
    AmpScaler,
    DistributedConfig,
    GPUDataConfig,
    GPUTrainer,
    GPUTrainingConfig,
    GPUTrainingResult,
    GPUTrainingState,
    ModelMemoryEstimate,
    estimate_training_memory,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson, LessonDataset, LessonIndex, LessonStore
from mopforge.runtime import (
    DeviceInfo,
    PrecisionPolicy,
    RuntimeConfig,
    RuntimeContext,
    build_runtime_context,
    detect_devices,
)
from mopforge.training import TinyTrainer, TrainerConfig, TrainerResult, TrainerState


@dataclass(frozen=True, slots=True)
class PublicAPIPolicy:
    """Machine-readable description of the public API contract."""

    stable: tuple[str, ...]
    experimental: tuple[str, ...]
    internal_prefixes: tuple[str, ...] = field(default_factory=lambda: ("_",))
    compatibility: str = (
        "Stable symbols are intended to remain import-compatible across minor "
        "0.x release-candidate updates. Experimental symbols may change while "
        "MoP-Forge moves toward v1.0-beta."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "stable": list(self.stable),
            "experimental": list(self.experimental),
            "internal_prefixes": list(self.internal_prefixes),
            "compatibility": self.compatibility,
        }


STABLE_PUBLIC_API = (
    "KnowledgeLesson",
    "LessonStore",
    "IndexedLessonStore",
    "LessonDataset",
    "LessonIndex",
    "MoPForgeConfig",
    "RuntimeConfig",
    "RuntimeContext",
    "DeviceInfo",
    "PrecisionPolicy",
    "build_runtime_context",
    "detect_devices",
    "TrainerConfig",
    "TrainerState",
    "TrainerResult",
    "TinyTrainer",
    "get_default_config",
    "list_default_config_names",
)

EXPERIMENTAL_PUBLIC_API = (
    "GPUTrainingConfig",
    "GPUTrainingState",
    "GPUTrainingResult",
    "GPUDataConfig",
    "GPUTrainer",
    "AmpScaler",
    "DistributedConfig",
    "ModelMemoryEstimate",
    "estimate_training_memory",
)

PUBLIC_API_POLICY = PublicAPIPolicy(
    stable=STABLE_PUBLIC_API,
    experimental=EXPERIMENTAL_PUBLIC_API,
)


__all__ = [
    "AmpScaler",
    "DeviceInfo",
    "DistributedConfig",
    "EXPERIMENTAL_PUBLIC_API",
    "GPUDataConfig",
    "GPUTrainer",
    "GPUTrainingConfig",
    "GPUTrainingResult",
    "GPUTrainingState",
    "IndexedLessonStore",
    "KnowledgeLesson",
    "LessonDataset",
    "LessonIndex",
    "LessonStore",
    "MoPForgeConfig",
    "ModelMemoryEstimate",
    "PUBLIC_API_POLICY",
    "PrecisionPolicy",
    "PublicAPIPolicy",
    "RuntimeConfig",
    "RuntimeContext",
    "STABLE_PUBLIC_API",
    "TinyTrainer",
    "TrainerConfig",
    "TrainerResult",
    "TrainerState",
    "build_runtime_context",
    "detect_devices",
    "estimate_training_memory",
    "get_default_config",
    "list_default_config_names",
]
