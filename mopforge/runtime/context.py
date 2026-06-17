"""Runtime context and movement helpers."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any

from mopforge.runtime.config import RuntimeConfig
from mopforge.runtime.devices import DeviceInfo, resolve_device
from mopforge.runtime.metadata import runtime_metadata
from mopforge.runtime.precision import (
    PrecisionPolicy,
    apply_tf32_policy,
    resolve_precision,
    torch_dtype_for_policy,
)


@dataclass(slots=True)
class RuntimeContext:
    """Resolved runtime state for one local run."""

    config: RuntimeConfig
    device_info: DeviceInfo
    precision_policy: PrecisionPolicy
    created_at: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "device_info": self.device_info.to_dict(),
            "precision_policy": self.precision_policy.to_dict(),
            "created_at": self.created_at,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def build_runtime_context(config: RuntimeConfig) -> RuntimeContext:
    """Resolve device/precision into a runtime context."""

    config = RuntimeConfig.from_dict(config.to_dict())
    device_info = resolve_device(config.device, config.require_device_available)
    precision = resolve_precision(
        config.precision,
        device_info,
        enable_amp=config.enable_amp,
        allow_tf32=config.allow_tf32,
    )
    warnings = []
    if device_info.reason:
        warnings.append(device_info.reason)
    warnings.extend(precision.warnings)
    warnings.extend(apply_tf32_policy(config.allow_tf32))
    return RuntimeContext(
        config=config,
        device_info=device_info,
        precision_policy=precision,
        created_at=datetime.now(timezone.utc).isoformat(),
        warnings=warnings,
    )


def move_batch_to_device(batch: Any, device: str) -> Any:
    """Recursively move torch tensors in ``batch`` to ``device``."""

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if is_dataclass(batch) and not isinstance(batch, type):
        try:
            values = {
                field.name: move_batch_to_device(getattr(batch, field.name), device)
                for field in fields(batch)
            }
            return replace(batch, **values)
        except Exception:
            return batch
    return batch


def move_model_to_runtime(model: Any, runtime: RuntimeContext) -> Any:
    """Move a model to the runtime device and optionally compile it."""

    moved = model
    to = getattr(moved, "to", None)
    if callable(to):
        moved = to(runtime.device_info.selected)
    if runtime.config.compile_model:
        try:
            import torch

            compiler = getattr(torch, "compile", None)
            if callable(compiler):
                moved = compiler(moved)
            else:
                runtime.warnings.append("compile_model requested but torch.compile is unavailable.")
        except Exception as exc:
            runtime.warnings.append(f"compile_model requested but failed: {exc}")
    return moved


def autocast_context(runtime: RuntimeContext):
    """Return an autocast context or ``nullcontext``."""

    if not runtime.precision_policy.amp_enabled:
        return contextlib.nullcontext()
    try:
        import torch
    except Exception:
        return contextlib.nullcontext()
    dtype = torch_dtype_for_policy(runtime.precision_policy)
    if dtype is None:
        return contextlib.nullcontext()
    try:
        return torch.autocast(
            device_type=runtime.precision_policy.autocast_device_type or runtime.device_info.device_type,
            dtype=dtype,
        )
    except Exception as exc:
        runtime.warnings.append(f"autocast unavailable for runtime: {exc}")
        return contextlib.nullcontext()


def apply_runtime_determinism(
    runtime: RuntimeContext,
    seed: int | None = None,
) -> list[str]:
    """Apply best-effort seed and deterministic controls."""

    notes: list[str] = []
    if seed is not None:
        random.seed(seed)
        notes.append("Seeded Python random.")
        try:
            import numpy as np

            np.random.seed(seed)
            notes.append("Seeded NumPy.")
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            notes.append("Seeded torch CPU.")
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                notes.append("Seeded torch CUDA.")
        except Exception as exc:
            notes.append(f"Could not seed torch: {exc}")
    if runtime.config.deterministic:
        try:
            import torch

            try:
                torch.use_deterministic_algorithms(True)
                notes.append("Enabled torch deterministic algorithms.")
            except Exception as exc:
                notes.append(f"Could not enable deterministic algorithms: {exc}")
            try:
                torch.backends.cudnn.benchmark = False
                notes.append("Disabled cuDNN benchmark mode.")
            except Exception:
                pass
        except Exception as exc:
            notes.append(f"Deterministic torch controls unavailable: {exc}")
    runtime.warnings.extend(note for note in notes if "Could not" in note or "unavailable" in note)
    return notes


__all__ = [
    "RuntimeContext",
    "apply_runtime_determinism",
    "autocast_context",
    "build_runtime_context",
    "move_batch_to_device",
    "move_model_to_runtime",
    "runtime_metadata",
]
