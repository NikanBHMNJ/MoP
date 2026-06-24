"""Approximate memory estimation for GPU job profiles."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelMemoryEstimate:
    parameter_count: int
    trainable_parameter_count: int
    weight_memory_gb: float
    gradient_memory_gb: float
    optimizer_memory_gb: float
    master_weight_memory_gb: float
    activation_memory_gb_estimate: float
    transient_memory_gb_estimate: float
    total_memory_gb_estimate: float
    fits: bool | None
    assumptions: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_training_memory(
    parameter_count: int,
    precision: str,
    optimizer: str = "adamw",
    micro_batch_size: int = 1,
    seq_len: int = 1024,
    d_model: int | None = None,
    n_layers: int | None = None,
    activation_checkpointing: bool = False,
    gpu_memory_gb: float | None = None,
    parameter_storage_precision: str = "fp32",
    gradient_storage_precision: str | None = None,
    optimizer_state_bytes_per_param: int = 8,
    master_weight_bytes_per_param: int = 0,
    transient_fraction: float = 0.1,
    trainable_parameter_count: int | None = None,
    distributed_shard_factor: int = 1,
) -> ModelMemoryEstimate:
    """Return a conservative approximate memory estimate.

    This is an inspectable planning heuristic, not a hardware guarantee.
    """

    if type(parameter_count) is not int or parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer.")
    if optimizer != "adamw":
        raise ValueError("Only adamw memory assumptions are implemented.")
    trainable_parameter_count = (
        parameter_count
        if trainable_parameter_count is None
        else int(trainable_parameter_count)
    )
    if trainable_parameter_count <= 0 or trainable_parameter_count > parameter_count:
        raise ValueError(
            "trainable_parameter_count must be positive and no greater than parameter_count."
        )
    if type(distributed_shard_factor) is not int or distributed_shard_factor <= 0:
        raise ValueError("distributed_shard_factor must be a positive integer.")
    activation_bytes = _bytes_per_param(precision)
    parameter_bytes = _bytes_per_param(parameter_storage_precision)
    gradient_bytes = _bytes_per_param(gradient_storage_precision or parameter_storage_precision)
    warnings: list[str] = []
    if precision == "fp8":
        warnings.append("FP8 execution is planning-only; estimating activations as bf16/fp16-sized.")
        activation_bytes = 2
    weights = parameter_count * parameter_bytes / distributed_shard_factor
    gradients = trainable_parameter_count * gradient_bytes / distributed_shard_factor
    optimizer_bytes = (
        trainable_parameter_count
        * optimizer_state_bytes_per_param
        / distributed_shard_factor
    )
    master_weights = (
        trainable_parameter_count
        * master_weight_bytes_per_param
        / distributed_shard_factor
    )
    d_model = int(d_model or 1024)
    n_layers = int(n_layers or max(1, parameter_count // max(d_model * d_model * 12, 1)))
    activation_factor = 2 if activation_checkpointing else 6
    activations = micro_batch_size * seq_len * d_model * n_layers * activation_bytes * activation_factor
    persistent = weights + gradients + optimizer_bytes + master_weights
    transient = int((persistent + activations) * max(0.0, float(transient_fraction)))
    total = persistent + activations + transient
    gb = 1024**3
    total_gb = total / gb
    fits = None if gpu_memory_gb is None else total_gb <= float(gpu_memory_gb) * 0.9
    if fits is False:
        warnings.append("Estimated memory exceeds 90% of target GPU memory.")
    return ModelMemoryEstimate(
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        weight_memory_gb=round(weights / gb, 4),
        gradient_memory_gb=round(gradients / gb, 4),
        optimizer_memory_gb=round(optimizer_bytes / gb, 4),
        master_weight_memory_gb=round(master_weights / gb, 4),
        activation_memory_gb_estimate=round(activations / gb, 4),
        transient_memory_gb_estimate=round(transient / gb, 4),
        total_memory_gb_estimate=round(total_gb, 4),
        fits=fits,
        assumptions={
            "precision": precision,
            "activation_bytes": activation_bytes,
            "parameter_storage_precision": parameter_storage_precision,
            "parameter_bytes": parameter_bytes,
            "gradient_storage_precision": gradient_storage_precision or parameter_storage_precision,
            "gradient_bytes": gradient_bytes,
            "optimizer": optimizer,
            "optimizer_bytes_per_param": optimizer_state_bytes_per_param,
            "master_weight_bytes_per_param": master_weight_bytes_per_param,
            "micro_batch_size": micro_batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_layers": n_layers,
            "activation_checkpointing": activation_checkpointing,
            "activation_factor": activation_factor,
            "fit_threshold_fraction": 0.9,
            "transient_fraction": transient_fraction,
            "note": "Autocast compute precision does not change parameter storage dtype.",
            "optimizer_and_gradient_scope": "trainable_parameters_only",
            "distributed_shard_factor": distributed_shard_factor,
            "distributed_storage_note": (
                "Parameter, gradient, and optimizer storage are divided by the "
                "declared shard factor; activations and transient workspace remain per rank."
            ),
        },
        warnings=warnings,
    )


def estimate_from_config(config) -> ModelMemoryEstimate:
    parameter_count = int(config.metadata.get("parameter_count", _rough_parameter_count(config)))
    trainable_parameter_count = int(
        config.metadata.get("trainable_parameter_count", parameter_count)
    )
    gpu_memory_gb = config.metadata.get("target_gpu_memory_gb")
    return estimate_training_memory(
        parameter_count=parameter_count,
        precision=config.precision if config.precision != "auto" else "bf16",
        optimizer=config.optimizer,
        micro_batch_size=config.micro_batch_size,
        seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        activation_checkpointing=config.activation_checkpointing,
        gpu_memory_gb=float(gpu_memory_gb) if gpu_memory_gb is not None else None,
        parameter_storage_precision=str(
            config.metadata.get("parameter_storage_precision", "fp32")
        ),
        gradient_storage_precision=str(
            config.metadata.get(
                "gradient_storage_precision",
                config.metadata.get("parameter_storage_precision", "fp32"),
            )
        ),
        optimizer_state_bytes_per_param=int(
            config.metadata.get("optimizer_state_bytes_per_param", 8)
        ),
        master_weight_bytes_per_param=int(
            config.metadata.get("master_weight_bytes_per_param", 0)
        ),
        transient_fraction=float(config.metadata.get("memory_transient_fraction", 0.1)),
        trainable_parameter_count=trainable_parameter_count,
        distributed_shard_factor=int(
            config.metadata.get("distributed_world_size", 1)
            if config.distributed_strategy == "fsdp"
            else 1
        ),
    )


def write_memory_estimate(estimate: ModelMemoryEstimate, path) -> str:
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(estimate.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(output)


def reset_cuda_peak_memory(runtime=None) -> None:
    """Reset CUDA peak memory stats when a CUDA runtime is active."""

    try:
        import torch

        if _is_cuda_runtime(runtime) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def cuda_memory_metrics(runtime=None) -> dict[str, Any]:
    """Return JSON-safe CUDA memory metrics in GiB, or nulls on non-CUDA paths."""

    payload: dict[str, float | None] = {
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "final_allocated_gb": None,
        "final_reserved_gb": None,
        "device_free_gb": None,
        "device_total_gb": None,
        "num_alloc_retries": None,
        "num_ooms": None,
        "inactive_split_gb": None,
        "allocator_cached_gb": None,
        "non_releasable_gb": None,
    }
    try:
        import torch

        if not (_is_cuda_runtime(runtime) and torch.cuda.is_available()):
            return payload
        gb = 1024**3
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        allocated = float(torch.cuda.memory_allocated())
        reserved = float(torch.cuda.memory_reserved())
        payload.update(
            {
                "peak_allocated_gb": round(float(torch.cuda.max_memory_allocated()) / gb, 4),
                "peak_reserved_gb": round(float(torch.cuda.max_memory_reserved()) / gb, 4),
                "final_allocated_gb": round(allocated / gb, 4),
                "final_reserved_gb": round(reserved / gb, 4),
                "device_free_gb": round(float(free_bytes) / gb, 4),
                "device_total_gb": round(float(total_bytes) / gb, 4),
                "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
                "num_ooms": int(stats.get("num_ooms", 0)),
                "inactive_split_gb": round(
                    float(stats.get("inactive_split_bytes.all.current", 0)) / gb,
                    4,
                ),
                "allocator_cached_gb": round(max(0.0, reserved - allocated) / gb, 4),
                "non_releasable_gb": round(
                    float(stats.get("inactive_split_bytes.all.current", 0)) / gb,
                    4,
                ),
            }
        )
    except Exception:
        pass
    return payload


def system_memory_metrics(path: str | Path = ".") -> dict[str, Any]:
    """Return best-effort host RAM and disk telemetry without hard dependencies."""

    payload: dict[str, Any] = {
        "host_rss_gb": None,
        "host_available_gb": None,
        "host_total_gb": None,
        "disk_free_gb": None,
        "disk_total_gb": None,
    }
    gb = 1024**3
    try:
        import psutil

        process = psutil.Process()
        vm = psutil.virtual_memory()
        payload.update(
            {
                "host_rss_gb": round(float(process.memory_info().rss) / gb, 4),
                "host_available_gb": round(float(vm.available) / gb, 4),
                "host_total_gb": round(float(vm.total) / gb, 4),
            }
        )
    except Exception:
        try:
            if Path("/proc/self/statm").exists():
                import os

                pages = int(Path("/proc/self/statm").read_text().split()[1])
                payload["host_rss_gb"] = round(
                    float(pages * os.sysconf("SC_PAGE_SIZE")) / gb,
                    4,
                )
            if Path("/proc/meminfo").exists():
                values = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0]) * 1024
                payload["host_total_gb"] = round(values["MemTotal"] / gb, 4)
                payload["host_available_gb"] = round(values["MemAvailable"] / gb, 4)
        except Exception:
            pass
    try:
        disk = shutil.disk_usage(Path(path).resolve())
        payload.update(
            {
                "disk_free_gb": round(float(disk.free) / gb, 4),
                "disk_total_gb": round(float(disk.total) / gb, 4),
            }
        )
    except Exception:
        pass
    return payload


def _bytes_per_param(precision: str) -> int:
    if precision in {"fp16", "bf16"}:
        return 2
    return 4


def _rough_parameter_count(config) -> int:
    vocab = int(config.metadata.get("vocab_size", 259))
    dense = vocab * config.d_model * 2
    blocks = config.n_layers * (12 * config.d_model * config.d_model)
    mop_extra = 0
    if config.model_type in {"mop_oracle", "mop_learned_router", "baseline_moe"}:
        module_count = max(4, len(config.target_modules or []))
        mop_extra = module_count * 4 * config.d_model * config.d_model
    return int(dense + blocks + mop_extra)


def _is_cuda_runtime(runtime) -> bool:
    if runtime is None:
        return False
    device_info = getattr(runtime, "device_info", None)
    return getattr(device_info, "device_type", None) == "cuda"
