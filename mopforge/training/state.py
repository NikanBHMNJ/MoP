"""Structured trainer configuration, state, and result schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mopforge.tokenization import TokenizerSpec
from mopforge.runtime import RuntimeConfig
from mopforge.training.parameter_policy import SUPPORTED_POLICY_MODES


@dataclass(slots=True)
class TrainerConfig:
    """CPU-first configuration for the reusable tiny trainer skeleton."""

    run_name: str = "tiny_trainer_run"
    seed: int = 123

    model_type: str = "dense"
    routing_mode: str = "none"
    model_ref: str | None = None

    lesson_path: str = "data/indexed_lessons.jsonl"
    index_path: str = "data/kts_index.sqlite"
    dataset_ref: str | None = None
    dataset_split: str | None = None
    dataset_version_id: str | None = None
    feedback_store_path: str | None = None
    queue_path: str | None = None

    tokenizer_type: str = "byte"
    tokenizer_name_or_path: str | None = None
    tokenizer_spec_path: str | None = None

    curriculum_strategy: str = "balanced"
    target_modules: list[str] | None = None
    curriculum_domains: list[str] | None = None
    curriculum_skills: list[str] | None = None
    curriculum_verification_statuses: list[str] | None = None

    trainable_policy_mode: str = "all"
    trainable_target_modules: list[str] | None = None
    train_router: bool = False
    train_embeddings: bool = False
    train_lm_head: bool = False
    train_shared_core: bool = True
    train_fast_adapters: bool = False
    train_generated_params: bool = False

    use_fast_adapters: bool = False
    fast_adapter_names: list[str] | None = None
    fast_adapter_bottleneck_dim: int = 16
    active_adapters: list[str] | None = None
    adapter_from_target_modules: bool = True

    use_generated_params: bool = False
    generated_condition_names: list[str] | None = None
    generated_condition_dim: int = 32
    generated_rank: int = 4
    generated_type: str = "low_rank_adapter"
    active_conditions: list[str] | None = None
    conditions_from_target_modules: bool = True

    batch_size: int = 2
    max_steps: int = 3
    eval_interval: int = 1
    checkpoint_interval: int = 1
    eval_batches: int = 1

    max_seq_len: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2

    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    device: str = "cpu"
    use_amp: bool = False
    precision: str = "fp32"
    enable_amp: bool = False
    allow_tf32: bool = False
    deterministic: bool = False
    compile_model: bool = False
    require_device_available: bool = True

    run_registry_root: str = "runs"
    artifact_root: str = "artifacts"
    save_checkpoints: bool = True
    resume_from: str | None = None
    save_full_checkpoints: bool = True
    resume_from_checkpoint: str | None = None
    checkpoint_every_steps: int | None = None
    save_rng_state: bool = True
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    training_kind: str = "trainer"
    source_config_kind: str | None = None
    source_config: dict[str, Any] | None = None

    run_generation_eval: bool = False
    generation_eval_examples: int = 2
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        """Validate CPU-safe trainer settings."""

        if self.model_type not in {"dense", "mop_oracle", "mop_learned_router"}:
            raise ValueError("model_type must be dense, mop_oracle, or mop_learned_router.")
        if self.routing_mode not in {"none", "oracle", "learned_router"}:
            raise ValueError("routing_mode must be none, oracle, or learned_router.")
        expected_routing = {
            "dense": "none",
            "mop_oracle": "oracle",
            "mop_learned_router": "learned_router",
        }[self.model_type]
        if self.routing_mode == "none" and self.model_type != "dense":
            self.routing_mode = expected_routing
        if self.routing_mode != expected_routing:
            raise ValueError(f"routing_mode for {self.model_type} must be {expected_routing}.")
        for field_name in (
            "batch_size",
            "max_steps",
            "eval_interval",
            "checkpoint_interval",
            "eval_batches",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.tokenizer_spec_path is not None:
            if (
                not isinstance(self.tokenizer_spec_path, str)
                or not self.tokenizer_spec_path.strip()
            ):
                raise ValueError("tokenizer_spec_path must be a non-empty string or None.")
            self.tokenizer_spec_path = self.tokenizer_spec_path.strip()
        if self.tokenizer_name_or_path is not None:
            if (
                not isinstance(self.tokenizer_name_or_path, str)
                or not self.tokenizer_name_or_path.strip()
            ):
                raise ValueError(
                    "tokenizer_name_or_path must be a non-empty string or None."
                )
            self.tokenizer_name_or_path = self.tokenizer_name_or_path.strip()
        if not self.tokenizer_spec_path:
            TokenizerSpec(
                tokenizer_type=self.tokenizer_type,
                name_or_path=self.tokenizer_name_or_path,
            )
        if self.trainable_policy_mode not in SUPPORTED_POLICY_MODES:
            valid = ", ".join(sorted(SUPPORTED_POLICY_MODES))
            raise ValueError(f"trainable_policy_mode must be one of: {valid}.")
        if self.trainable_target_modules is not None:
            if isinstance(self.trainable_target_modules, str):
                raise ValueError("trainable_target_modules must be a list of strings.")
            if not all(
                isinstance(module, str) and module.strip()
                for module in self.trainable_target_modules
            ):
                raise ValueError("trainable_target_modules must contain non-empty strings.")
            seen = set()
            self.trainable_target_modules = [
                module
                for module in self.trainable_target_modules
                if not (module in seen or seen.add(module))
            ]
        for field_name in (
            "target_modules",
            "curriculum_domains",
            "curriculum_skills",
            "curriculum_verification_statuses",
        ):
            values = getattr(self, field_name)
            if values is None:
                continue
            if isinstance(values, str):
                raise ValueError(f"{field_name} must be a list of strings.")
            if not all(isinstance(value, str) and value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings.")
            seen = set()
            setattr(
                self,
                field_name,
                [
                    value
                    for value in values
                    if not (value in seen or seen.add(value))
                ],
            )
        if type(self.fast_adapter_bottleneck_dim) is not int or self.fast_adapter_bottleneck_dim <= 0:
            raise ValueError("fast_adapter_bottleneck_dim must be a positive integer.")
        if type(self.generated_condition_dim) is not int or self.generated_condition_dim <= 0:
            raise ValueError("generated_condition_dim must be a positive integer.")
        if type(self.generated_rank) is not int or self.generated_rank <= 0:
            raise ValueError("generated_rank must be a positive integer.")
        if self.generated_type not in {"low_rank_adapter", "scale_shift"}:
            raise ValueError("generated_type must be low_rank_adapter or scale_shift.")
        for field_name in (
            "fast_adapter_names",
            "active_adapters",
            "generated_condition_names",
            "active_conditions",
        ):
            values = getattr(self, field_name)
            if values is None:
                continue
            if isinstance(values, str):
                raise ValueError(f"{field_name} must be a list of strings.")
            if not all(isinstance(name, str) and name.strip() for name in values):
                raise ValueError(f"{field_name} must contain non-empty strings.")
            seen = set()
            setattr(
                self,
                field_name,
                [
                    name
                    for name in values
                    if not (name in seen or seen.add(name))
                ],
            )
        if self.trainable_policy_mode == "fast_adapters_only":
            self.train_fast_adapters = True
        if self.trainable_policy_mode == "generated_params_only":
            self.train_generated_params = True
        if self.use_fast_adapters and self.fast_adapter_names is None:
            self.fast_adapter_names = ["default"]
        if self.use_generated_params and self.generated_condition_names is None:
            self.generated_condition_names = ["default"]
        self.enable_amp = bool(self.enable_amp or self.use_amp)
        self.use_amp = bool(self.enable_amp)
        RuntimeConfig(
            device=self.device,
            precision=self.precision,
            enable_amp=self.enable_amp,
            allow_tf32=self.allow_tf32,
            deterministic=self.deterministic,
            compile_model=self.compile_model,
            require_device_available=self.require_device_available,
        )
        if self.checkpoint_every_steps is not None and (
            type(self.checkpoint_every_steps) is not int
            or self.checkpoint_every_steps <= 0
        ):
            raise ValueError("checkpoint_every_steps must be a positive integer or None.")
        if self.resume_from_checkpoint is not None:
            if (
                not isinstance(self.resume_from_checkpoint, str)
                or not self.resume_from_checkpoint.strip()
            ):
                raise ValueError("resume_from_checkpoint must be a non-empty string or None.")
            self.resume_from_checkpoint = self.resume_from_checkpoint.strip()
        if self.resume_from is not None:
            if not isinstance(self.resume_from, str) or not self.resume_from.strip():
                raise ValueError("resume_from must be a non-empty string or None.")
            self.resume_from = self.resume_from.strip()
        if self.training_kind not in {"trainer", "sft", "pretrain"}:
            raise ValueError("training_kind must be trainer, sft, or pretrain.")
        if self.source_config_kind is not None:
            if (
                not isinstance(self.source_config_kind, str)
                or not self.source_config_kind.strip()
            ):
                raise ValueError("source_config_kind must be a non-empty string or None.")
            self.source_config_kind = self.source_config_kind.strip()
        if self.source_config is not None and not isinstance(self.source_config, dict):
            raise ValueError("source_config must be a dictionary or None.")
        for field_name in ("model_ref", "dataset_ref", "dataset_split", "dataset_version_id"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string or None.")
            setattr(self, field_name, value.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return asdict(self)


@dataclass(slots=True)
class TrainerState:
    """Mutable tiny trainer progress state."""

    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float | None = None
    latest_train_loss: float | None = None
    latest_eval_loss: float | None = None
    checkpoint_artifacts: list[str] = field(default_factory=list)
    full_checkpoint_artifacts: list[str] = field(default_factory=list)
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    parameter_counts: dict[str, int] = field(default_factory=dict)
    parameter_group_summaries: list[dict[str, Any]] = field(default_factory=list)
    resume_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable state dictionary."""

        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "latest_train_loss": self.latest_train_loss,
            "latest_eval_loss": self.latest_eval_loss,
            "checkpoint_artifacts": list(self.checkpoint_artifacts),
            "full_checkpoint_artifacts": list(self.full_checkpoint_artifacts),
            "metrics_history": [dict(item) for item in self.metrics_history],
            "parameter_counts": dict(self.parameter_counts),
            "parameter_group_summaries": [
                dict(item) for item in self.parameter_group_summaries
            ],
            "resume_metadata": dict(self.resume_metadata),
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerState":
        """Create trainer state from a dictionary."""

        return cls(
            global_step=int(data.get("global_step", 0)),
            epoch=int(data.get("epoch", 0)),
            best_eval_loss=data.get("best_eval_loss"),
            latest_train_loss=data.get("latest_train_loss"),
            latest_eval_loss=data.get("latest_eval_loss"),
            checkpoint_artifacts=list(data.get("checkpoint_artifacts", [])),
            full_checkpoint_artifacts=list(data.get("full_checkpoint_artifacts", [])),
            metrics_history=[dict(item) for item in data.get("metrics_history", [])],
            parameter_counts=dict(data.get("parameter_counts", {})),
            parameter_group_summaries=[
                dict(item) for item in data.get("parameter_group_summaries", [])
            ],
            resume_metadata=dict(data.get("resume_metadata", {})),
            runtime_metadata=dict(data.get("runtime_metadata", {})),
        )


@dataclass(slots=True)
class TrainerResult:
    """Final result for one tiny trainer run."""

    run_id: str
    run_name: str
    model_type: str
    routing_mode: str
    final_state: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "model_type": self.model_type,
            "routing_mode": self.routing_mode,
            "final_state": dict(self.final_state),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "finite": bool(self.finite),
        }

    def save_json(self, path: str | Path) -> Path:
        """Write this trainer result to JSON and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path
