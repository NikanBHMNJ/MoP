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
    token_shard_manifest: str | None = None
    tokenizer_type: str = "byte"
    tokenizer_name_or_path: str | None = None
    tokenizer_spec_path: str | None = None
    tokenizer_vocab_size: int | None = None

    output_root: str = "gpu_runs"
    artifact_root: str = "artifacts"
    run_id: str | None = None

    max_steps: int = 100
    max_optimizer_steps: int | None = None
    max_train_tokens: int | None = None
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_every_steps: int = 50
    eval_every_optimizer_steps: int | None = None
    eval_batches: int = 2
    eval_full_dataset: bool = False
    save_every_steps: int = 100
    save_every_optimizer_steps: int | None = None
    log_every_steps: int = 10
    log_every_optimizer_steps: int | None = None
    shuffle_train: bool = True
    train_shuffle_seed: int = 42

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    optimizer: str = "adamw"
    scheduler: str = "none"
    warmup_steps: int = 0
    warmup_optimizer_steps: int | None = None
    scheduler_unit: str = "optimizer_steps"
    warmup_tokens: int = 0
    min_lr_ratio: float = 0.0
    early_stopping_enabled: bool = False
    early_stopping_patience_evals: int = 5
    early_stopping_min_delta: float = 0.0

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    architecture_family: str = "tiny_transformer"
    intermediate_size: int | None = None
    n_key_value_heads: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
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
    distributed_strategy: str = "none"
    distributed_backend: str = "nccl"
    distributed_timeout_seconds: int = 1800
    fsdp_use_orig_params: bool = True
    fsdp_cpu_offload: bool = False
    distributed_checkpoint_mode: str = "full"

    activation_checkpointing: bool = False
    efficient_attention: str = "auto"
    empty_cache_every_steps: int | None = None
    empty_cache_every_optimizer_steps: int | None = None

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
    generation_eval_use_best_checkpoint: bool = True
    generation_eval_include_train: bool = False
    generation_eval_stratify_by: str | None = None

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
        if self.architecture_family not in {"tiny_transformer", "production_decoder_v2"}:
            raise ValueError(
                "architecture_family must be tiny_transformer or production_decoder_v2."
            )
        if self.intermediate_size is not None and (
            type(self.intermediate_size) is not int or self.intermediate_size <= 0
        ):
            raise ValueError("intermediate_size must be a positive integer or None.")
        if self.n_key_value_heads is not None:
            if type(self.n_key_value_heads) is not int or self.n_key_value_heads <= 0:
                raise ValueError("n_key_value_heads must be a positive integer or None.")
            if self.n_heads % self.n_key_value_heads:
                raise ValueError("n_heads must be divisible by n_key_value_heads.")
        for field_name in ("rope_theta", "rms_norm_eps"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        for field_name in ("dropout", "attention_dropout"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{field_name} must be in [0.0, 1.0).")
            setattr(self, field_name, float(value))
        if type(self.train_shuffle_seed) is not int:
            raise ValueError("train_shuffle_seed must be an integer.")
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
        if self.max_train_tokens is not None and (
            type(self.max_train_tokens) is not int or self.max_train_tokens <= 0
        ):
            raise ValueError("max_train_tokens must be a positive integer or None.")
        if self.scheduler_unit not in {"optimizer_steps", "tokens"}:
            raise ValueError("scheduler_unit must be optimizer_steps or tokens.")
        if type(self.warmup_tokens) is not int or self.warmup_tokens < 0:
            raise ValueError("warmup_tokens must be a non-negative integer.")
        if self.scheduler_unit == "tokens" and self.max_train_tokens is None:
            raise ValueError("Token-unit scheduling requires max_train_tokens.")
        if self.max_train_tokens is not None and self.warmup_tokens >= self.max_train_tokens:
            raise ValueError("warmup_tokens must be smaller than max_train_tokens.")
        if not isinstance(self.min_lr_ratio, (int, float)) or not 0.0 <= float(self.min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1].")
        self.min_lr_ratio = float(self.min_lr_ratio)
        for field_name in (
            "max_optimizer_steps",
            "eval_every_optimizer_steps",
            "save_every_optimizer_steps",
            "log_every_optimizer_steps",
            "warmup_optimizer_steps",
            "empty_cache_every_optimizer_steps",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field_name} must be a positive integer or None.")
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
        if self.generation_eval_stratify_by not in {None, "bug_type", "domain", "skill"}:
            raise ValueError(
                "generation_eval_stratify_by must be bug_type, domain, skill, or None."
            )
        for field_name in ("max_train_examples", "max_eval_examples", "prefetch_factor", "empty_cache_every_steps"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field_name} must be a positive integer or None.")
        if not isinstance(self.tokenizer_type, str):
            raise ValueError("tokenizer_type must be byte or hf.")
        self.tokenizer_type = self.tokenizer_type.strip().lower()
        if self.tokenizer_type not in {"byte", "hf"}:
            raise ValueError("tokenizer_type must be byte or hf.")
        if self.tokenizer_type == "hf" and not (
            self.tokenizer_name_or_path or self.tokenizer_spec_path
        ):
            raise ValueError(
                "HF tokenizer configs require tokenizer_name_or_path or tokenizer_spec_path."
            )
        if self.tokenizer_vocab_size is not None and (
            type(self.tokenizer_vocab_size) is not int or self.tokenizer_vocab_size <= 0
        ):
            raise ValueError("tokenizer_vocab_size must be a positive integer or None.")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        if self.distributed_strategy not in {"none", "ddp", "fsdp"}:
            raise ValueError("distributed_strategy must be none, ddp, or fsdp.")
        if self.distributed_backend not in {"nccl", "gloo"}:
            raise ValueError("distributed_backend must be nccl or gloo.")
        if type(self.distributed_timeout_seconds) is not int or self.distributed_timeout_seconds <= 0:
            raise ValueError("distributed_timeout_seconds must be positive.")
        if self.distributed_checkpoint_mode not in {"full", "sharded"}:
            raise ValueError("distributed_checkpoint_mode must be full or sharded.")
        if self.distributed_strategy == "fsdp" and self.architecture_family != "production_decoder_v2":
            raise ValueError("FSDP currently requires architecture_family=production_decoder_v2.")
        if self.distributed_strategy == "fsdp" and self.distributed_checkpoint_mode != "sharded":
            raise ValueError("FSDP requires distributed_checkpoint_mode=sharded.")
        if self.distributed_strategy != "none" and self.activation_cache_path:
            raise ValueError(
                "Distributed cached-tail training is not supported; run the offloaded sparse tail on one GPU."
            )
        if self.distributed_strategy != "none" and self.run_generation_eval:
            raise ValueError(
                "Distributed generated-code evaluation is not supported during training; "
                "consolidate the checkpoint and run `mopforge eval code` afterward."
            )
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
            "generation_eval_use_best_checkpoint",
            "generation_eval_include_train",
            "early_stopping_enabled",
            "eval_full_dataset",
            "shuffle_train",
            "tie_word_embeddings",
            "fsdp_use_orig_params",
            "fsdp_cpu_offload",
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
            "token_shard_manifest",
            "tokenizer_name_or_path",
            "tokenizer_spec_path",
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

    @property
    def optimizer_step_budget(self) -> int:
        """Return the exact optimizer-update budget for this run."""

        if self.max_optimizer_steps is not None:
            return int(self.max_optimizer_steps)
        return int(
            (self.max_steps + self.gradient_accumulation_steps - 1)
            // self.gradient_accumulation_steps
        )

    @property
    def microstep_budget(self) -> int:
        """Return the microstep budget, preserving legacy ``max_steps`` jobs."""

        if self.max_optimizer_steps is not None:
            return int(self.max_optimizer_steps * self.gradient_accumulation_steps)
        return int(self.max_steps)

    @property
    def scheduler_warmup_optimizer_steps(self) -> int:
        """Resolve scheduler warmup in optimizer updates.

        ``warmup_steps`` historically advanced once per optimizer update despite
        its ambiguous name. Keep that behavior for old configs.
        """

        if self.warmup_optimizer_steps is not None:
            return int(self.warmup_optimizer_steps)
        return int(self.warmup_steps)

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
    train_epoch: int = 0
    train_batches_in_epoch: int = 0
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
            train_epoch=int(data.get("train_epoch", 0)),
            train_batches_in_epoch=int(data.get("train_batches_in_epoch", 0)),
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
