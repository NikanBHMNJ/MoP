"""Runtime/device/precision foundation for MoP-Forge."""

from mopforge.runtime.config import RuntimeConfig, runtime_config_from_kwargs
from mopforge.runtime.context import (
    RuntimeContext,
    apply_runtime_determinism,
    autocast_context,
    build_runtime_context,
    move_batch_to_device,
    move_model_to_runtime,
)
from mopforge.runtime.devices import DeviceInfo, detect_devices, resolve_device
from mopforge.runtime.metadata import runtime_metadata
from mopforge.runtime.precision import (
    PrecisionPolicy,
    apply_tf32_policy,
    resolve_precision,
    torch_dtype_for_policy,
)

__all__ = [
    "DeviceInfo",
    "PrecisionPolicy",
    "RuntimeConfig",
    "RuntimeContext",
    "apply_runtime_determinism",
    "apply_tf32_policy",
    "autocast_context",
    "build_runtime_context",
    "detect_devices",
    "move_batch_to_device",
    "move_model_to_runtime",
    "resolve_device",
    "resolve_precision",
    "runtime_config_from_kwargs",
    "runtime_metadata",
    "torch_dtype_for_policy",
]
