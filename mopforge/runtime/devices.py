"""Device detection and resolution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mopforge.runtime.config import RuntimeConfig


@dataclass(slots=True)
class DeviceInfo:
    """Resolved device metadata."""

    requested: str
    selected: str
    device_type: str
    available: bool
    reason: str = ""
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    cuda_device_count: int = 0
    gpu_name: str | None = None
    gpu_memory_total_gb: float | None = None
    capability: str | None = None
    mps_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return asdict(self)


def detect_devices() -> dict[str, Any]:
    """Return JSON-safe CPU/CUDA/MPS inventory."""

    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "cpu_available": True,
    }
    try:
        import torch
    except Exception:
        return info

    info["torch_available"] = True
    info["torch_version"] = str(getattr(torch, "__version__", "unknown"))
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            info["cuda_available"] = bool(cuda.is_available())
        except Exception:
            info["cuda_available"] = False
        info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            count = int(cuda.device_count()) if info["cuda_available"] else 0
        except Exception:
            count = 0
        info["cuda_device_count"] = count
        devices = []
        for index in range(count):
            devices.append(_cuda_device_payload(torch, index))
        info["cuda_devices"] = devices
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None:
        try:
            info["mps_available"] = bool(mps.is_available())
        except Exception:
            info["mps_available"] = False
    return info


def resolve_device(
    requested: str = "cpu",
    require_available: bool = True,
) -> DeviceInfo:
    """Resolve a requested device into a selected device."""

    config = RuntimeConfig(device=requested, require_device_available=require_available)
    requested = config.device
    detected = detect_devices()

    if requested == "cpu":
        return _info(requested, "cpu", "cpu", True, "CPU is always available.", detected)
    if requested == "auto":
        if detected["cuda_available"] and detected["cuda_device_count"] > 0:
            return _info(requested, "cuda:0", "cuda", True, "Auto selected CUDA.", detected)
        if detected["mps_available"]:
            return _info(requested, "mps", "mps", True, "Auto selected MPS.", detected)
        return _info(requested, "cpu", "cpu", True, "Auto selected CPU.", detected)
    if requested == "mps":
        if detected["mps_available"]:
            return _info(requested, "mps", "mps", True, "MPS is available.", detected)
        return _unavailable(requested, "MPS requested but unavailable.", require_available, detected)
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda:"):
        return _resolve_cuda(requested, require_available, detected)
    raise RuntimeError(f"Unsupported device request: {requested}")


def _resolve_cuda(requested: str, require_available: bool, detected: dict[str, Any]) -> DeviceInfo:
    if not detected["cuda_available"]:
        return _unavailable(requested, "CUDA requested but CUDA is unavailable.", require_available, detected)
    try:
        index = int(requested.split(":", 1)[1])
    except Exception:
        index = 0
    count = int(detected.get("cuda_device_count") or 0)
    if index < 0 or index >= count:
        return _unavailable(
            requested,
            f"CUDA device index {index} is out of range for {count} device(s).",
            require_available,
            detected,
        )
    device = detected.get("cuda_devices", [{}])[index]
    return _info(
        requested,
        f"cuda:{index}",
        "cuda",
        True,
        "CUDA device is available.",
        detected,
        gpu_name=device.get("name"),
        gpu_memory_total_gb=device.get("total_memory_gb"),
        capability=device.get("capability"),
    )


def _unavailable(
    requested: str,
    reason: str,
    require_available: bool,
    detected: dict[str, Any],
) -> DeviceInfo:
    if require_available:
        raise RuntimeError(reason)
    return _info(requested, "cpu", "cpu", False, f"{reason} Falling back to CPU.", detected)


def _info(
    requested: str,
    selected: str,
    device_type: str,
    available: bool,
    reason: str,
    detected: dict[str, Any],
    *,
    gpu_name: str | None = None,
    gpu_memory_total_gb: float | None = None,
    capability: str | None = None,
) -> DeviceInfo:
    return DeviceInfo(
        requested=requested,
        selected=selected,
        device_type=device_type,
        available=available,
        reason=reason,
        torch_available=bool(detected.get("torch_available")),
        torch_version=detected.get("torch_version"),
        cuda_available=bool(detected.get("cuda_available")),
        cuda_version=detected.get("cuda_version"),
        cuda_device_count=int(detected.get("cuda_device_count") or 0),
        gpu_name=gpu_name,
        gpu_memory_total_gb=gpu_memory_total_gb,
        capability=capability,
        mps_available=bool(detected.get("mps_available")),
        metadata={"detected": detected},
    )


def _cuda_device_payload(torch, index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"index": index, "name": None, "total_memory_gb": None, "capability": None}
    try:
        payload["name"] = str(torch.cuda.get_device_name(index))
    except Exception:
        pass
    try:
        props = torch.cuda.get_device_properties(index)
        payload["total_memory_gb"] = round(float(props.total_memory) / (1024**3), 3)
    except Exception:
        pass
    try:
        major, minor = torch.cuda.get_device_capability(index)
        payload["capability"] = f"{major}.{minor}"
    except Exception:
        pass
    return payload
