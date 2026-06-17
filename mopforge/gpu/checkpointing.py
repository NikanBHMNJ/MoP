"""Checkpoint helpers for GPUTrainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mopforge.lifecycle import capture_rng_state, restore_rng_state


def save_gpu_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    state=None,
    config=None,
    runtime_metadata=None,
    data_metadata=None,
    model_metadata=None,
    memory_metadata=None,
) -> str:
    """Save a full GPUTrainer checkpoint and return the path."""

    torch = _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = state.to_dict() if hasattr(state, "to_dict") else dict(state or {})
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    payload = {
        "checkpoint_format": "mopforge_gpu_train_v1",
        "training_kind": "gpu_train",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else {},
        "trainer_state": state_dict,
        "config": config_dict,
        "runtime_metadata": dict(runtime_metadata or {}),
        "data_metadata": dict(data_metadata or {}),
        "model_metadata": dict(model_metadata or {}),
        "memory_metadata": dict(memory_metadata or {}),
        "rng_state": capture_rng_state(),
        "metadata": {
            "training_kind": "gpu_train",
            "global_step": state_dict.get("global_step"),
            "optimizer_step": state_dict.get("optimizer_step"),
            "tokens_seen": state_dict.get("tokens_seen"),
            "precision": (runtime_metadata or {}).get("selected_precision"),
            "device": (runtime_metadata or {}).get("selected_device"),
            "amp_enabled": (runtime_metadata or {}).get("amp_enabled"),
            "scaler_enabled": bool((scaler.state_dict() if scaler is not None else {}).get("enabled")),
        },
    }
    torch.save(payload, output)
    return str(output)


def load_gpu_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    torch = _require_torch()
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def restore_gpu_checkpoint(
    payload: dict[str, Any],
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore model/optimizer/scheduler/scaler state best-effort."""

    metadata: dict[str, Any] = {"restored": []}
    model.load_state_dict(payload["model_state"])
    metadata["restored"].append("model")
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        metadata["restored"].append("optimizer")
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
        metadata["restored"].append("scheduler")
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload.get("scaler_state", {}))
        metadata["restored"].append("scaler")
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
        metadata["restored"].append("rng")
    return metadata


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GPU checkpointing.") from exc
    return torch
