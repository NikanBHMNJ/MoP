"""Distributed sharded model/optimizer checkpoint helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from mopforge.gpu.distributed import DistributedRuntime, distributed_barrier


def save_sharded_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    trainer_state=None,
    config=None,
    runtime: DistributedRuntime | None = None,
    metadata=None,
) -> str:
    """Collectively save sharded model and optimizer state with DCP."""

    torch = _require_torch()
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    distributed = runtime or DistributedRuntime()
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=options,
    )
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=output / "shards",
        process_group=(
            torch.distributed.group.WORLD
            if distributed.enabled and torch.distributed.is_initialized()
            else None
        ),
        no_dist=not distributed.enabled,
    )
    if distributed.is_primary:
        sidecar = {
            "checkpoint_format": "mopforge_distributed_sharded_v1",
            "trainer_state": _to_dict(trainer_state),
            "config": _to_dict(config),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "distributed": distributed.to_dict(),
            "metadata": dict(metadata or {}),
        }
        _atomic_torch_save(sidecar, output / "metadata.pt")
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint_format": sidecar["checkpoint_format"],
                    "metadata_path": "metadata.pt",
                    "shard_path": "shards",
                    "world_size": distributed.world_size,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    distributed_barrier(distributed)
    return str(output)


def load_sharded_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    runtime: DistributedRuntime | None = None,
) -> dict:
    """Collectively restore model/optimizer and return trainer metadata."""

    torch = _require_torch()
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    candidate = Path(path)
    distributed = runtime or DistributedRuntime()
    sidecar = _torch_load(candidate / "metadata.pt")
    if sidecar.get("checkpoint_format") != "mopforge_distributed_sharded_v1":
        raise ValueError(f"Unsupported distributed checkpoint: {candidate}")
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(
        state,
        checkpoint_id=candidate / "shards",
        process_group=(
            torch.distributed.group.WORLD
            if distributed.enabled and torch.distributed.is_initialized()
            else None
        ),
        no_dist=not distributed.enabled,
    )
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    if scheduler is not None and sidecar.get("scheduler_state") is not None:
        scheduler.load_state_dict(sidecar["scheduler_state"])
    if scaler is not None and sidecar.get("scaler_state") is not None:
        scaler.load_state_dict(sidecar["scaler_state"])
    distributed_barrier(distributed)
    return sidecar


def is_sharded_checkpoint(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / "manifest.json").is_file()


def _atomic_torch_save(payload, path: Path) -> None:
    torch = _require_torch()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Path):
    torch = _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_dict(value):
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for distributed checkpoints.") from exc
    return torch
