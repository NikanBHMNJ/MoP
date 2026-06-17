"""FT/SFT mode configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mopforge.models import (
    adapter_names_from_target_modules,
    condition_names_from_target_modules,
)
from mopforge.runtime import RuntimeConfig
from mopforge.sft.modes import get_training_mode_spec


@dataclass(slots=True)
class FinetuneConfig:
    """CPU-smoke fine-tuning configuration."""

    mode: str = "sft_full"
    model_type: str = "dense"
    model_ref: str | None = None
    target_modules: list[str] | None = None

    lesson_path: str = "data/indexed_lessons.jsonl"
    index_path: str = "data/kts_index.sqlite"
    dataset_ref: str | None = None
    dataset_split: str | None = None
    dataset_version_id: str | None = None
    feedback_store_path: str | None = None

    tokenizer_type: str = "byte"
    tokenizer_name_or_path: str | None = None
    tokenizer_spec_path: str | None = None

    curriculum_strategy: str = "balanced"
    skill_filter: str | None = None
    domain_filter: str | None = "coding"
    verification_status_filter: str | None = None

    use_fast_adapters: bool = False
    fast_adapter_names: list[str] | None = None
    adapter_from_target_modules: bool = True
    use_generated_params: bool = False
    generated_condition_names: list[str] | None = None
    generated_rank: int = 4
    generated_type: str = "low_rank_adapter"

    batch_size: int = 2
    max_steps: int = 3
    eval_batches: int = 1
    max_seq_len: int = 512
    learning_rate: float = 1e-3
    device: str = "cpu"
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
    seed: int = 123

    def __post_init__(self) -> None:
        """Validate and normalize FT/SFT settings."""

        spec = get_training_mode_spec(self.mode)
        if self.model_type not in {"dense", "mop_oracle", "mop_learned_router"}:
            raise ValueError("model_type must be dense, mop_oracle, or mop_learned_router.")
        self.target_modules = _normalize_strings(self.target_modules, "target_modules")
        self.fast_adapter_names = _normalize_strings(
            self.fast_adapter_names,
            "fast_adapter_names",
        )
        self.generated_condition_names = _normalize_strings(
            self.generated_condition_names,
            "generated_condition_names",
        )

        if spec.requires_target_modules and not self.target_modules:
            raise ValueError(f"{self.mode} requires target_modules.")
        if self.mode == "sft_module":
            self.model_type = "mop_oracle"
        elif self.mode == "sft_adapter":
            self.model_type = "mop_oracle" if self.model_type == "dense" else self.model_type
            self.use_fast_adapters = True
            if self.fast_adapter_names is None:
                derived = adapter_names_from_target_modules(self.target_modules or [])
                self.fast_adapter_names = derived or ["default"]
        elif self.mode == "sft_generated":
            self.model_type = "mop_oracle" if self.model_type == "dense" else self.model_type
            self.use_generated_params = True
            if self.generated_condition_names is None:
                derived = condition_names_from_target_modules(self.target_modules or [])
                self.generated_condition_names = derived or ["default"]
        elif self.mode == "sft_router":
            self.model_type = "mop_learned_router"
        elif self.mode == "repair_sft":
            if self.curriculum_strategy == "balanced":
                self.curriculum_strategy = "repair_boosted"
            if self.skill_filter is None and self.verification_status_filter is None:
                self.skill_filter = "repair"

        for field_name in ("batch_size", "max_steps", "eval_batches", "max_seq_len"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        RuntimeConfig(
            device=self.device,
            precision=self.precision,
            enable_amp=self.enable_amp,
            allow_tf32=self.allow_tf32,
            deterministic=self.deterministic,
            compile_model=self.compile_model,
            require_device_available=self.require_device_available,
        )
        if type(self.generated_rank) is not int or self.generated_rank <= 0:
            raise ValueError("generated_rank must be a positive integer.")
        if self.generated_type not in {"low_rank_adapter", "scale_shift"}:
            raise ValueError("generated_type must be low_rank_adapter or scale_shift.")
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


def _normalize_strings(values: list[str] | None, field_name: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a list of strings.")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    seen = set()
    return [value for value in values if not (value in seen or seen.add(value))]
