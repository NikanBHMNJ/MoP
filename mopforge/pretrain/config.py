"""Configuration and result schemas for continued-pretraining smoke runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mopforge.runtime import RuntimeConfig
from mopforge.training.parameter_policy import SUPPORTED_POLICY_MODES


@dataclass(slots=True)
class ContinuedPretrainConfig:
    """CPU-safe continued-pretraining smoke configuration."""

    run_name: str = "tiny_continued_pretrain"
    seed: int = 123
    corpus_path: str = "data/text_corpus.jsonl"
    corpus_dataset_ref: str | None = None
    dataset_split: str | None = None
    lesson_path: str | None = "data/indexed_lessons.jsonl"
    index_path: str | None = "data/kts_index.sqlite"
    tokenizer_type: str = "byte"
    tokenizer_name_or_path: str | None = None
    tokenizer_spec_path: str | None = None
    model_type: str = "dense"
    model_ref: str | None = None
    curriculum_strategy: str = "sequential"
    batch_size: int = 2
    max_steps: int = 3
    eval_batches: int = 1
    max_seq_len: int = 512
    stride: int | None = None
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
    save_full_checkpoints: bool = True
    resume_from_checkpoint: str | None = None
    checkpoint_every_steps: int | None = None
    save_rng_state: bool = True
    save_optimizer_state: bool = True
    use_fast_adapters: bool = False
    fast_adapter_names: list[str] | None = None
    use_generated_params: bool = False
    generated_condition_names: list[str] | None = None
    generated_condition_dim: int = 32
    generated_rank: int = 4
    generated_type: str = "low_rank_adapter"
    trainable_policy_mode: str = "all"

    def __post_init__(self) -> None:
        """Validate CPU-safe settings."""

        if self.model_type not in {"dense", "mop_oracle", "mop_learned_router"}:
            raise ValueError("model_type must be dense, mop_oracle, or mop_learned_router.")
        for field_name in (
            "batch_size",
            "max_steps",
            "eval_batches",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.stride is not None and (type(self.stride) is not int or self.stride <= 0):
            raise ValueError("stride must be a positive integer or None.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not isinstance(self.tokenizer_type, str) or not self.tokenizer_type.strip():
            raise ValueError("tokenizer_type must be a non-empty string.")
        self.tokenizer_type = self.tokenizer_type.strip().lower()
        if self.tokenizer_name_or_path is not None:
            if (
                not isinstance(self.tokenizer_name_or_path, str)
                or not self.tokenizer_name_or_path.strip()
            ):
                raise ValueError(
                    "tokenizer_name_or_path must be a non-empty string or None."
                )
            self.tokenizer_name_or_path = self.tokenizer_name_or_path.strip()
        if self.tokenizer_spec_path is not None:
            if (
                not isinstance(self.tokenizer_spec_path, str)
                or not self.tokenizer_spec_path.strip()
            ):
                raise ValueError("tokenizer_spec_path must be a non-empty string or None.")
            self.tokenizer_spec_path = self.tokenizer_spec_path.strip()
        if self.tokenizer_type == "hf" and self.tokenizer_name_or_path is None and self.tokenizer_spec_path is None:
            raise ValueError("HF tokenizer configs require tokenizer_name_or_path or tokenizer_spec_path.")
        if self.trainable_policy_mode not in SUPPORTED_POLICY_MODES:
            valid = ", ".join(sorted(SUPPORTED_POLICY_MODES))
            raise ValueError(f"trainable_policy_mode must be one of: {valid}.")
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
        if type(self.generated_condition_dim) is not int or self.generated_condition_dim <= 0:
            raise ValueError("generated_condition_dim must be a positive integer.")
        if type(self.generated_rank) is not int or self.generated_rank <= 0:
            raise ValueError("generated_rank must be a positive integer.")
        if self.generated_type not in {"low_rank_adapter", "scale_shift"}:
            raise ValueError("generated_type must be low_rank_adapter or scale_shift.")
        if self.fast_adapter_names is not None:
            if isinstance(self.fast_adapter_names, str):
                raise ValueError("fast_adapter_names must be a list of strings.")
            if not all(isinstance(name, str) and name.strip() for name in self.fast_adapter_names):
                raise ValueError("fast_adapter_names must contain non-empty strings.")
            seen = set()
            self.fast_adapter_names = [
                name for name in self.fast_adapter_names
                if not (name in seen or seen.add(name))
            ]
        if self.use_fast_adapters and self.fast_adapter_names is None:
            self.fast_adapter_names = ["default"]
        if self.generated_condition_names is not None:
            if isinstance(self.generated_condition_names, str):
                raise ValueError("generated_condition_names must be a list of strings.")
            if not all(isinstance(name, str) and name.strip() for name in self.generated_condition_names):
                raise ValueError("generated_condition_names must contain non-empty strings.")
            seen = set()
            self.generated_condition_names = [
                name for name in self.generated_condition_names
                if not (name in seen or seen.add(name))
            ]
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
        for field_name in ("model_ref", "corpus_dataset_ref", "dataset_split"):
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
class ContinuedPretrainResult:
    """Result for one continued-pretraining smoke run."""

    run_id: str
    run_name: str
    model_type: str
    corpus_records: int
    corpus_chunks: int
    final_train_loss: float | None
    final_eval_loss: float | None
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "model_type": self.model_type,
            "corpus_records": self.corpus_records,
            "corpus_chunks": self.corpus_chunks,
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "finite": bool(self.finite),
        }

    def save_json(self, path: str | Path) -> Path:
        """Write this result to JSON and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path
