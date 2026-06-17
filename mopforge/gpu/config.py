"""GPU training configuration and result schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mopforge.runtime import RuntimeConfig
from mopforge.training.parameter_policy import SUPPORTED_POLICY_MODES


MODEL_TYPES = {"dense", "mop_oracle", "mop_learned_router", "baseline_moe"}
OPTIMIZERS = {"adamw"}
SCHEDULERS = {"none", "cosine", "linear_warmup"}
EFFICIENT_ATTENTION = {"auto", "torch_sdpa", "eager"}


@dataclass(slots=True)
class GPUTrainingConfig:
    """Single-device GPU-aware training config with CPU-safe fallback."""

    name: str = "gpu-train"
    model_type: str = "dense"
    model_ref: str | None = None
    dataset_ref: str | None = None
    dataset_split: str | None = None
    lesson_path: str = "data/coding_bugfix_lessons.jsonl"
    index_path: str | None = None
    corpus_path: str | None = None

    output_root: str = "gpu_runs"
    artifact_root: str = "artifacts"
    run_id: str | None = None

    max_steps: int = 100
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_every_steps: int = 50
    eval_batches: int = 2
    save_every_steps: int = 100
    log_every_steps: int = 10

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    optimizer: str = "adamw"
    scheduler: str = "none"
    warmup_steps: int = 0

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 1024

    trainable_policy_mode: str = "all"
    target_modules: list[str] = field(default_factory=list)

    use_fast_adapters: bool = False
    fast_adapter_names: list[str] | None = None
    fast_adapter_bottleneck_dim: int = 16

    use_generated_params: bool = False
    generated_condition_names: list[str] | None = None
    generated_condition_dim: int = 16
    generated_rank: int = 4
    generated_type: str = "low_rank_adapter"

    device: str = "auto"
    precision: str = "auto"
    enable_amp: bool = True
    allow_tf32: bool = True
    deterministic: bool = False
    compile_model: bool = False
    require_device_available: bool = True

    activation_checkpointing: bool = False
    efficient_attention: str = "auto"
    empty_cache_every_steps: int | None = None

    save_full_checkpoints: bool = True
    resume_from_checkpoint: str | None = None
    save_optimizer_state: bool = True
    save_rng_state: bool = True

    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _non_empty(self.name, "name")
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(sorted(MODEL_TYPES))}.")
        for field_name in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "eval_every_steps",
            "eval_batches",
            "save_every_steps",
            "log_every_steps",
            "d_model",
            "n_layers",
            "n_heads",
            "max_seq_len",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None.")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of: {', '.join(sorted(OPTIMIZERS))}.")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of: {', '.join(sorted(SCHEDULERS))}.")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer.")
        if self.trainable_policy_mode not in SUPPORTED_POLICY_MODES:
            raise ValueError("trainable_policy_mode is not supported.")
        if self.efficient_attention not in EFFICIENT_ATTENTION:
            raise ValueError("efficient_attention must be auto, torch_sdpa, or eager.")
        for field_name in (
            "target_modules",
            "fast_adapter_names",
            "generated_condition_names",
        ):
            setattr(self, field_name, _optional_strings(getattr(self, field_name), field_name))
        if type(self.fast_adapter_bottleneck_dim) is not int or self.fast_adapter_bottleneck_dim <= 0:
            raise ValueError("fast_adapter_bottleneck_dim must be a positive integer.")
        if type(self.generated_condition_dim) is not int or self.generated_condition_dim <= 0:
            raise ValueError("generated_condition_dim must be a positive integer.")
        if type(self.generated_rank) is not int or self.generated_rank <= 0:
            raise ValueError("generated_rank must be a positive integer.")
        if self.generated_type not in {"low_rank_adapter", "scale_shift"}:
            raise ValueError("generated_type must be low_rank_adapter or scale_shift.")
        for field_name in ("max_train_examples", "max_eval_examples", "prefetch_factor", "empty_cache_every_steps"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field_name} must be a positive integer or None.")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        for field_name in (
            "use_fast_adapters",
            "use_generated_params",
            "enable_amp",
            "allow_tf32",
            "deterministic",
            "compile_model",
            "require_device_available",
            "activation_checkpointing",
            "save_full_checkpoints",
            "save_optimizer_state",
            "save_rng_state",
            "pin_memory",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")
        for field_name in (
            "model_ref",
            "dataset_ref",
            "dataset_split",
            "lesson_path",
            "index_path",
            "corpus_path",
            "output_root",
            "artifact_root",
            "run_id",
            "resume_from_checkpoint",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, _non_empty(value, field_name))
        RuntimeConfig(
            device=self.device,
            precision=self.precision,
            enable_amp=self.enable_amp,
            allow_tf32=self.allow_tf32,
            deterministic=self.deterministic,
            compile_model=self.compile_model,
            require_device_available=self.require_device_available,
        )
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        json.dumps(self.metadata)

    @property
    def effective_batch_size(self) -> int:
        return int(self.micro_batch_size * self.gradient_accumulation_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUTrainingConfig":
        if not isinstance(data, dict):
            raise TypeError("GPUTrainingConfig.from_dict expects a dictionary.")
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "GPUTrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class GPUTrainingState:
    global_step: int = 0
    optimizer_step: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0
    latest_train_loss: float | None = None
    latest_eval_loss: float | None = None
    best_eval_loss: float | None = None
    latest_checkpoint_path: str | None = None
    scaler_state: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    metric_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUTrainingState":
        return cls(
            global_step=int(data.get("global_step", 0)),
            optimizer_step=int(data.get("optimizer_step", 0)),
            samples_seen=int(data.get("samples_seen", 0)),
            tokens_seen=int(data.get("tokens_seen", 0)),
            latest_train_loss=data.get("latest_train_loss"),
            latest_eval_loss=data.get("latest_eval_loss"),
            best_eval_loss=data.get("best_eval_loss"),
            latest_checkpoint_path=data.get("latest_checkpoint_path"),
            scaler_state=dict(data.get("scaler_state", {})),
            runtime_metadata=dict(data.get("runtime_metadata", {})),
            memory_snapshots=[dict(item) for item in data.get("memory_snapshots", [])],
            metric_history=[dict(item) for item in data.get("metric_history", [])],
        )


@dataclass(slots=True)
class GPUTrainingResult:
    run_id: str
    status: str
    config: dict[str, Any]
    state: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    runtime_metadata: dict[str, Any]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output


def _positive_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_strings(values: list[str] | None, field_name: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a list of strings or None.")
    if not isinstance(values, list):
        values = list(values)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    seen = set()
    return [item.strip() for item in values if not (item.strip() in seen or seen.add(item.strip()))]
