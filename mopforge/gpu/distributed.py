"""Torchrun/distributed launch planning for GPU jobs."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DistributedConfig:
    strategy: str = "none"
    num_nodes: int = 1
    nproc_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    node_rank: int = 0
    backend: str = "nccl"
    dry_run: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "torchrun", "ddp_plan", "ddp", "fsdp"}:
            raise ValueError("strategy must be none, torchrun, ddp_plan, ddp, or fsdp.")
        if type(self.num_nodes) is not int or self.num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer.")
        if type(self.nproc_per_node) is not int or self.nproc_per_node <= 0:
            raise ValueError("nproc_per_node must be a positive integer.")
        if type(self.master_port) is not int or not 0 < self.master_port < 65536:
            raise ValueError("master_port must be a valid TCP port.")
        if type(self.node_rank) is not int or self.node_rank < 0:
            raise ValueError("node_rank must be a non-negative integer.")
        if self.backend not in {"nccl", "gloo"}:
            raise ValueError("backend must be nccl or gloo.")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be a boolean.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistributedConfig":
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "DistributedConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_torchrun_command(config_path: str, distributed: DistributedConfig | dict[str, Any] | None = None) -> list[str]:
    cfg = distributed if isinstance(distributed, DistributedConfig) else DistributedConfig.from_dict(distributed or {"strategy": "torchrun"})
    if cfg.strategy == "none":
        return ["mopforge", "gpu", "train", config_path]
    return [
        "torchrun",
        "--nnodes",
        str(cfg.num_nodes),
        "--nproc_per_node",
        str(cfg.nproc_per_node),
        "--master_addr",
        cfg.master_addr,
        "--master_port",
        str(cfg.master_port),
        "--node_rank",
        str(cfg.node_rank),
        "-m",
        "mopforge.cli.main",
        "gpu",
        "train",
        config_path,
    ]


def validate_distributed_plan(distributed: DistributedConfig, execute: bool = False) -> list[str]:
    messages: list[str] = []
    if execute and distributed.backend == "nccl":
        try:
            import torch

            if not torch.cuda.is_available():
                messages.append("ERROR: nccl execution requires CUDA.")
        except Exception:
            messages.append("ERROR: nccl execution requires PyTorch CUDA.")
    if distributed.strategy in {"torchrun", "ddp_plan"} and distributed.dry_run:
        messages.append("WARNING: torchrun launcher is in dry-run planning mode.")
    return messages


@dataclass(slots=True)
class DistributedRuntime:
    strategy: str = "none"
    backend: str = "gloo"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized_here: bool = False
    wrapped: bool = False
    wrapper: str | None = None

    @property
    def enabled(self) -> bool:
        return self.strategy in {"ddp", "fsdp"} and self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initialize_distributed_runtime(
    strategy: str,
    *,
    backend: str = "nccl",
    timeout_seconds: int = 1800,
) -> DistributedRuntime:
    """Initialize torch.distributed from torchrun environment variables."""

    if strategy not in {"none", "ddp", "fsdp"}:
        raise ValueError("Training distributed strategy must be none, ddp, or fsdp.")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    runtime = DistributedRuntime(
        strategy=strategy,
        backend=backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    if strategy == "none" or world_size <= 1:
        return runtime
    torch = _require_torch()
    import datetime

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=int(timeout_seconds)),
        )
        runtime.initialized_here = True
    return runtime


def wrap_distributed_model(
    model,
    runtime: DistributedRuntime,
    *,
    precision: str = "bf16",
    fsdp_use_orig_params: bool = True,
    fsdp_cpu_offload: bool = False,
):
    """Wrap a model with DDP or FSDP after trainable policy application."""

    if not runtime.enabled:
        return model
    torch = _require_torch()
    if runtime.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        wrapped = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if torch.cuda.is_available() else None,
            output_device=runtime.local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        runtime.wrapped = True
        runtime.wrapper = "DistributedDataParallel"
        return wrapped
    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from mopforge.models.production_decoder import ProductionDecoderBlock

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(precision, torch.bfloat16)
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    wrapped = FullyShardedDataParallel(
        model,
        auto_wrap_policy=ModuleWrapPolicy({ProductionDecoderBlock}),
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=bool(fsdp_cpu_offload)),
        device_id=runtime.local_rank,
        sync_module_states=True,
        use_orig_params=bool(fsdp_use_orig_params),
        limit_all_gathers=True,
    )
    runtime.wrapped = True
    runtime.wrapper = "FullyShardedDataParallel"
    return wrapped


def distributed_no_sync(model, runtime: DistributedRuntime, *, synchronize: bool):
    if runtime.enabled and not synchronize and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def distributed_barrier(runtime: DistributedRuntime | None) -> None:
    if runtime is None or not runtime.enabled:
        return
    torch = _require_torch()
    torch.distributed.barrier()


def distributed_sum(value: float | int, runtime: DistributedRuntime | None) -> float:
    if runtime is None or not runtime.enabled:
        return float(value)
    torch = _require_torch()
    device = (
        torch.device(f"cuda:{runtime.local_rank}")
        if runtime.backend == "nccl"
        else torch.device("cpu")
    )
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def broadcast_object(value, runtime: DistributedRuntime | None):
    if runtime is None or not runtime.enabled:
        return value
    torch = _require_torch()
    values = [value if runtime.is_primary else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def finalize_distributed_runtime(runtime: DistributedRuntime | None) -> None:
    if runtime is None or not runtime.initialized_here:
        return
    torch = _require_torch()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for distributed training.") from exc
    return torch
