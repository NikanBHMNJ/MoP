"""Runtime metadata helpers."""

from __future__ import annotations

from typing import Any


def runtime_metadata(runtime) -> dict[str, Any]:
    """Return JSON-safe metadata for a runtime context."""

    device = runtime.device_info
    precision = runtime.precision_policy
    return {
        "requested_device": device.requested,
        "selected_device": device.selected,
        "device_type": device.device_type,
        "device_available": bool(device.available),
        "torch_available": bool(device.torch_available),
        "torch_version": device.torch_version,
        "cuda_available": bool(device.cuda_available),
        "cuda_device_count": int(device.cuda_device_count),
        "gpu_name": device.gpu_name,
        "requested_precision": precision.requested,
        "selected_precision": precision.selected,
        "torch_dtype_name": precision.torch_dtype_name,
        "amp_enabled": bool(precision.amp_enabled),
        "allow_tf32": bool(precision.allow_tf32),
        "compile_model": bool(runtime.config.compile_model),
        "deterministic": bool(runtime.config.deterministic),
        "warnings": list(runtime.warnings),
    }
