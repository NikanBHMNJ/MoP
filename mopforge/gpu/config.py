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
MOP_BLOCK_TYPES = {"post_core_mlp", "routed_ffn"}
ROUTING_GRANULARITIES = {"example", "token"}
ACTIVATION_CACHE_DTYPES = {"fp32", "fp16", "bf16"}


@dataclass(slots=True)
class GPUTrainingConfig:
    """Single-device GPU-aware training config with CPU-safe fallback."""

    name: str = "gpu-train"
    model_type: str = "dense"
    model_ref: str | None = None
    dataset_ref: str | None = None
    dataset_split: str | None = None
    dataset_split_id: str | None = None
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
    early_stopping_enabled: bool = False
    early_stopping_patience_evals: int = 5
    early_stopping_min_delta: float = 0.0

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 1024
    module_names: list[str] | None = None
    always_include_core: bool = True
    mop_block_type: str = "post_core_mlp"
    expert_count: int | None = None
    active_experts: int = 1
    routing_granularity: str = "example"
    shared_depth_ratio: float = 1.0
    use_lora_deltas: bool = False
    lora_tail_only: bool = False
    lora_rank: int = 0
    lora_target_modules: list[str] | None = None

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
    save_best_eval_checkpoint: bool = True
    resume_from_checkpoint: str | None = None
    resume_model_only: bool = False
    save_trainable_only_checkpoints: bool = False
    base_checkpoint_path: str | None = None
    save_optimizer_state: bool = True
    save_rng_state: bool = True

    activation_cache_path: str | None = None
    activation_cache_dtype: str = "bf16"
    offload_frozen_backbone_for_cache: bool = True
    distillation_enabled: bool = False
    distillation_weight: float = 0.0
    distillation_temperature: float = 1.0
    distillation_top_k: int = 0
    hard_example_replay_enabled: bool = False
    hard_example_replay_loss_threshold: float | None = None
    hard_example_replay_multiplier: int = 1
    target_eval_loss: float | None = None
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int | None = None
    run_generation_eval: bool = False
    generation_eval_examples: int = 2
    generation_max_new_tokens: int = 32

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
            "active_experts",
            "generation_eval_examples",
            "generation_max_new_tokens",
            "early_stopping_patience_evals",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.distillation_weight < 0:
            raise ValueError("distillation_weight must be non-negative.")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive.")
        if type(self.distillation_top_k) is not int or self.distillation_top_k < 0:
            raise ValueError("distillation_top_k must be a non-negative integer.")
        if type(self.hard_example_replay_multiplier) is not int or self.hard_example_replay_multiplier <= 0:
            raise ValueError("hard_example_replay_multiplier must be a positive integer.")
        if self.hard_example_replay_loss_threshold is not None:
            if (
                not isinstance(self.hard_example_replay_loss_threshold, (int, float))
                or self.hard_example_replay_loss_threshold < 0
            ):
                raise ValueError("hard_example_replay_loss_threshold must be non-negative or None.")
            self.hard_example_replay_loss_threshold = float(self.hard_example_replay_loss_threshold)
        if self.target_eval_loss is not None:
            if not isinstance(self.target_eval_loss, (int, float)) or self.target_eval_loss <= 0:
                raise ValueError("target_eval_loss must be a positive number or None.")
            self.target_eval_loss = float(self.target_eval_loss)
        if not isinstance(self.early_stopping_min_delta, (int, float)) or self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be a non-negative number.")
        self.early_stopping_min_delta = float(self.early_stopping_min_delta)
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
        if self.mop_block_type not in MOP_BLOCK_TYPES:
            raise ValueError("mop_block_type must be post_core_mlp or routed_ffn.")
        if self.routing_granularity not in ROUTING_GRANULARITIES:
            raise ValueError("routing_granularity must be example or token.")
        if self.activation_cache_dtype not in ACTIVATION_CACHE_DTYPES:
            raise ValueError("activation_cache_dtype must be fp32, fp16, or bf16.")
        if self.expert_count is not None and (type(self.expert_count) is not int or self.expert_count <= 0):
            raise ValueError("expert_count must be a positive integer or None.")
        if type(self.shared_depth_ratio) not in {float, int} or not 0.0 < float(self.shared_depth_ratio) <= 1.0:
            raise ValueError("shared_depth_ratio must be in (0.0, 1.0].")
        self.shared_depth_ratio = float(self.shared_depth_ratio)
        if type(self.lora_rank) is not int or self.lora_rank < 0:
            raise ValueError("lora_rank must be a non-negative integer.")
        if self.use_lora_deltas and self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive when use_lora_deltas is true.")
        if not isinstance(self.lora_tail_only, bool):
            raise ValueError("lora_tail_only must be a boolean.")
        if self.lora_tail_only and not self.use_lora_deltas:
            raise ValueError("lora_tail_only requires use_lora_deltas=true.")
        for field_name in (
            "module_names",
            "target_modules",
            "fast_adapter_names",
            "generated_condition_names",
            "lora_target_modules",
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
            "always_include_core",
            "save_full_checkpoints",
            "save_best_eval_checkpoint",
            "resume_model_only",
            "save_trainable_only_checkpoints",
            "save_optimizer_state",
            "save_rng_state",
            "use_lora_deltas",
            "lora_tail_only",
            "offload_frozen_backbone_for_cache",
            "distillation_enabled",
            "hard_example_replay_enabled",
            "pin_memory",
            "run_generation_eval",
            "early_stopping_enabled",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")
        for field_name in (
            "model_ref",
            "dataset_ref",
            "dataset_split",
            "dataset_split_id",
            "lesson_path",
            "index_path",
            "corpus_path",
            "output_root",
            "artifact_root",
            "run_id",
            "resume_from_checkpoint",
            "base_checkpoint_path",
            "activation_cache_path",
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
    evals_without_improvement: int = 0
    target_eval_loss_reached: bool = False
    target_eval_loss_value: float | None = None
    target_eval_loss_step: int | None = None
    target_eval_loss_samples_seen: int | None = None
    target_eval_loss_tokens_seen: int | None = None
    target_eval_loss_time_sec: float | None = None
    target_eval_loss_memory_snapshot: dict[str, Any] = field(default_factory=dict)
    stopped_early: bool = False
    stop_reason: str | None = None
    latest_checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None
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
            evals_without_improvement=int(data.get("evals_without_improvement", 0)),
            target_eval_loss_reached=bool(data.get("target_eval_loss_reached", False)),
            target_eval_loss_value=data.get("target_eval_loss_value"),
            target_eval_loss_step=data.get("target_eval_loss_step"),
            target_eval_loss_samples_seen=data.get("target_eval_loss_samples_seen"),
            target_eval_loss_tokens_seen=data.get("target_eval_loss_tokens_seen"),
            target_eval_loss_time_sec=data.get("target_eval_loss_time_sec"),
            target_eval_loss_memory_snapshot=dict(data.get("target_eval_loss_memory_snapshot") or {}),
            stopped_early=bool(data.get("stopped_early", False)),
            stop_reason=data.get("stop_reason"),
            latest_checkpoint_path=data.get("latest_checkpoint_path"),
            best_checkpoint_path=data.get("best_checkpoint_path"),
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
