"""Precision and AMP policy helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mopforge.runtime.config import RuntimeConfig
from mopforge.runtime.devices import DeviceInfo


@dataclass(slots=True)
class PrecisionPolicy:
    """Resolved precision policy."""

    requested: str
    selected: str
    torch_dtype_name: str
    amp_enabled: bool
    autocast_device_type: str | None = None
    allow_tf32: bool = False
    fp8_requested: bool = False
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_precision(
    precision: str,
    device_info: DeviceInfo,
    enable_amp: bool = False,
    allow_tf32: bool = False,
) -> PrecisionPolicy:
    """Resolve requested precision for a resolved device."""

    RuntimeConfig(precision=precision)
    requested = precision.strip().lower()
    warnings: list[str] = []
    device_type = device_info.device_type
    cuda_bf16 = _cuda_bf16_supported(device_info)

    if requested == "fp8":
        warnings.append("FP8 execution is planning-only; falling back to a safe precision.")
        if device_type == "cuda":
            selected = "bf16" if cuda_bf16 else "fp16"
        else:
            selected = "fp32"
        return _policy(requested, selected, device_type, enable_amp, allow_tf32, warnings, fp8=True)

    if requested == "auto":
        if device_type == "cuda":
            selected = "bf16" if cuda_bf16 else "fp16"
        else:
            selected = "fp32"
        return _policy(requested, selected, device_type, enable_amp, allow_tf32, warnings)

    if requested == "fp32":
        return _policy(requested, "fp32", device_type, False, allow_tf32, warnings)

    if requested in {"fp16", "bf16"}:
        if device_type == "cuda":
            return _policy(requested, requested, device_type, enable_amp, allow_tf32, warnings)
        warnings.append(f"{requested} execution is not enabled for {device_type}; falling back to fp32.")
        return _policy(requested, "fp32", device_type, False, allow_tf32, warnings)

    raise ValueError(f"Unsupported precision: {precision}")


def apply_tf32_policy(allow_tf32: bool) -> list[str]:
    """Apply best-effort TF32 backend settings."""

    notes: list[str] = []
    if not allow_tf32:
        return notes
    try:
        import torch
    except Exception:
        return ["TF32 requested but PyTorch is not installed."]
    try:
        if not torch.cuda.is_available():
            return ["TF32 requested but CUDA is not available."]
    except Exception:
        return ["TF32 requested but CUDA availability could not be checked."]
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        notes.append("Enabled torch.backends.cuda.matmul.allow_tf32.")
    except Exception as exc:
        notes.append(f"Could not enable CUDA matmul TF32: {exc}")
    try:
        torch.backends.cudnn.allow_tf32 = True
        notes.append("Enabled torch.backends.cudnn.allow_tf32.")
    except Exception as exc:
        notes.append(f"Could not enable cuDNN TF32: {exc}")
    return notes


def torch_dtype_for_policy(policy: PrecisionPolicy):
    """Return a torch dtype object for a precision policy, or ``None``."""

    try:
        import torch
    except Exception:
        return None
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(policy.selected)


def _policy(
    requested: str,
    selected: str,
    device_type: str,
    enable_amp: bool,
    allow_tf32: bool,
    warnings: list[str],
    *,
    fp8: bool = False,
) -> PrecisionPolicy:
    amp = bool(enable_amp and device_type == "cuda" and selected in {"fp16", "bf16"})
    if enable_amp and not amp:
        warnings.append("AMP requested but disabled for this device/precision.")
    return PrecisionPolicy(
        requested=requested,
        selected=selected,
        torch_dtype_name={
            "fp32": "torch.float32",
            "fp16": "torch.float16",
            "bf16": "torch.bfloat16",
        }.get(selected, "torch.float32"),
        amp_enabled=amp,
        autocast_device_type=device_type if amp else None,
        allow_tf32=bool(allow_tf32),
        fp8_requested=fp8,
        warnings=list(warnings),
    )


def _cuda_bf16_supported(device_info: DeviceInfo) -> bool:
    if device_info.device_type != "cuda" or not device_info.cuda_available:
        return False
    try:
        import torch

        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker):
            return bool(checker())
    except Exception:
        return False
    return False
