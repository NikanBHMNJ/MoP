"""Approximate memory estimation for GPU job profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelMemoryEstimate:
    parameter_count: int
    weight_memory_gb: float
    gradient_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb_estimate: float
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
) -> ModelMemoryEstimate:
    """Return a conservative approximate memory estimate.

    This is an inspectable planning heuristic, not a hardware guarantee.
    """

    if type(parameter_count) is not int or parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer.")
    if optimizer != "adamw":
        raise ValueError("Only adamw memory assumptions are implemented.")
    bytes_per_param = _bytes_per_param(precision)
    warnings: list[str] = []
    if precision == "fp8":
        warnings.append("FP8 execution is planning-only; estimating weights as bf16/fp16-sized.")
        bytes_per_param = 2
    weights = parameter_count * bytes_per_param
    gradients = parameter_count * bytes_per_param
    optimizer_bytes = parameter_count * 8
    d_model = int(d_model or 1024)
    n_layers = int(n_layers or max(1, parameter_count // max(d_model * d_model * 12, 1)))
    activation_factor = 2 if activation_checkpointing else 6
    activations = micro_batch_size * seq_len * d_model * n_layers * bytes_per_param * activation_factor
    total = weights + gradients + optimizer_bytes + activations
    gb = 1024**3
    total_gb = total / gb
    fits = None if gpu_memory_gb is None else total_gb <= float(gpu_memory_gb) * 0.9
    if fits is False:
        warnings.append("Estimated memory exceeds 90% of target GPU memory.")
    return ModelMemoryEstimate(
        parameter_count=parameter_count,
        weight_memory_gb=round(weights / gb, 4),
        gradient_memory_gb=round(gradients / gb, 4),
        optimizer_memory_gb=round(optimizer_bytes / gb, 4),
        activation_memory_gb_estimate=round(activations / gb, 4),
        total_memory_gb_estimate=round(total_gb, 4),
        fits=fits,
        assumptions={
            "precision": precision,
            "bytes_per_param": bytes_per_param,
            "optimizer": optimizer,
            "optimizer_bytes_per_param": 8,
            "micro_batch_size": micro_batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_layers": n_layers,
            "activation_checkpointing": activation_checkpointing,
            "activation_factor": activation_factor,
            "fit_threshold_fraction": 0.9,
        },
        warnings=warnings,
    )


def estimate_from_config(config) -> ModelMemoryEstimate:
    parameter_count = int(config.metadata.get("parameter_count", _rough_parameter_count(config)))
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
    )


def write_memory_estimate(estimate: ModelMemoryEstimate, path) -> str:
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(estimate.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(output)


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
