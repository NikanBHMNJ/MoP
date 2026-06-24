"""Checkpoint helpers for GPUTrainer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
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
    trainable_only: bool = False,
    base_checkpoint_path: str | None = None,
    trainable_policy: dict[str, Any] | None = None,
) -> str:
    """Save a full GPUTrainer checkpoint and return the path."""

    torch = _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = state.to_dict() if hasattr(state, "to_dict") else dict(state or {})
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    architecture_dict = dict((model_metadata or {}).get("architecture") or {})
    config_sha256 = _stable_sha256(config_dict)
    architecture_sha256 = _stable_sha256(architecture_dict) if architecture_dict else None
    activation_cache_path = (data_metadata or {}).get("source_path")
    activation_cache_metadata = dict((data_metadata or {}).get("cache_metadata") or {})
    model_state = None if trainable_only else model.state_dict()
    trainable_model_state = _trainable_state_dict(model) if trainable_only else None
    payload = {
        "checkpoint_format": (
            "mopforge_gpu_train_sparse_v1"
            if trainable_only
            else "mopforge_gpu_train_v1"
        ),
        "training_kind": "gpu_train",
        "model_state": model_state,
        "trainable_model_state": trainable_model_state,
        "trainable_parameter_names": (
            sorted(trainable_model_state)
            if trainable_model_state is not None
            else None
        ),
        "base_checkpoint_path": base_checkpoint_path,
        "config_sha256": config_sha256,
        "architecture_sha256": architecture_sha256,
        "trainable_policy": dict(trainable_policy or {}),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else {},
        "trainer_state": state_dict,
        "config": config_dict,
        "runtime_metadata": dict(runtime_metadata or {}),
        "data_metadata": dict(data_metadata or {}),
        "model_metadata": dict(model_metadata or {}),
        "memory_metadata": dict(memory_metadata or {}),
        "activation_cache_path": activation_cache_path,
        "activation_cache_metadata": activation_cache_metadata,
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
            "trainable_only": bool(trainable_only),
            "base_checkpoint_path": base_checkpoint_path,
            "activation_cache_path": activation_cache_path,
            "activation_cache_manifest_sha256": activation_cache_metadata.get("manifest_sha256"),
            "config_sha256": config_sha256,
            "architecture_sha256": architecture_sha256,
        },
    }
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
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
    restore_optimizer: bool = True,
    restore_scheduler: bool = True,
    restore_scaler: bool = True,
    strict_model: bool = True,
) -> dict[str, Any]:
    """Restore model/optimizer/scheduler/scaler state best-effort."""

    metadata: dict[str, Any] = {"restored": [], "skipped": []}
    base_checkpoint = payload.get("base_checkpoint_path")
    if payload.get("trainable_model_state") is not None and base_checkpoint:
        base_path = Path(base_checkpoint)
        if base_path.exists():
            base_payload = load_gpu_checkpoint(base_path)
            base_metadata = restore_gpu_checkpoint(
                base_payload,
                model=model,
                restore_rng=False,
                restore_optimizer=False,
                restore_scheduler=False,
                restore_scaler=False,
                strict_model=False,
            )
            metadata["base_checkpoint"] = {
                "path": str(base_path),
                "metadata": base_metadata,
            }
        else:
            metadata["skipped"].append(f"base_checkpoint_missing:{base_checkpoint}")
    if payload.get("model_state") is not None:
        model_metadata = _load_model_state(
            model,
            payload["model_state"],
            strict_model=strict_model,
        )
        metadata.update(model_metadata)
        metadata["restored"].append("model")
    elif payload.get("trainable_model_state") is not None:
        model_metadata = _load_model_state(
            model,
            payload["trainable_model_state"],
            strict_model=False,
        )
        metadata.update(model_metadata)
        metadata["restored"].append("trainable_model_state")
    else:
        metadata["skipped"].append("model_state_missing")
    if restore_optimizer and optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        metadata["restored"].append("optimizer")
    if restore_scheduler and scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
        metadata["restored"].append("scheduler")
    if restore_scaler and scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload.get("scaler_state", {}))
        metadata["restored"].append("scaler")
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
        metadata["restored"].append("rng")
    return metadata


def _trainable_state_dict(model) -> dict[str, Any]:
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name in trainable_names
    }


def _load_model_state(model, state_dict: dict[str, Any], *, strict_model: bool) -> dict[str, Any]:
    if strict_model:
        model.load_state_dict(state_dict)
        return {
            "model_load_strict": True,
            "model_loaded_key_count": len(state_dict),
            "model_skipped_key_count": 0,
        }
    warm_start_metadata = None
    adapter = getattr(model, "adapt_warm_start_state_dict", None)
    if callable(adapter):
        state_dict = adapter(state_dict)
        warm_start_metadata = dict(
            getattr(model, "last_warm_start_metadata", {}) or {}
        )
    current = model.state_dict()
    compatible = {}
    skipped = []
    for name, tensor in state_dict.items():
        if name not in current:
            skipped.append({"name": name, "reason": "missing_in_model"})
            continue
        if tuple(current[name].shape) != tuple(tensor.shape):
            skipped.append(
                {
                    "name": name,
                    "reason": "shape_mismatch",
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(current[name].shape),
                }
            )
            continue
        compatible[name] = tensor
    result = model.load_state_dict(compatible, strict=False)
    metadata = {
        "model_load_strict": False,
        "model_loaded_key_count": len(compatible),
        "model_skipped_key_count": len(skipped),
        "model_skipped_keys": skipped[:50],
        "model_missing_key_count": len(getattr(result, "missing_keys", [])),
        "model_unexpected_key_count": len(getattr(result, "unexpected_keys", [])),
    }
    if warm_start_metadata:
        metadata["warm_start_adaptation"] = warm_start_metadata
    return metadata


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GPU checkpointing.") from exc
    return torch


def _stable_sha256(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
