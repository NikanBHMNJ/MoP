"""Builders and runner for FT/SFT mode API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopforge.models import (
    adapter_names_from_target_modules,
    condition_names_from_target_modules,
)
from mopforge.sft.config import FinetuneConfig
from mopforge.sft.modes import TrainingModeSpec, get_training_mode_spec
from mopforge.training import TinyTrainer, TrainerConfig


def build_finetune_lesson_filter(config: FinetuneConfig) -> dict[str, Any]:
    """Build deterministic curriculum filters for a fine-tune config."""

    filters: dict[str, Any] = {}
    if config.domain_filter is not None:
        filters["domains"] = [config.domain_filter]
    if config.skill_filter is not None:
        filters["skills"] = [config.skill_filter]
    if config.verification_status_filter is not None:
        filters["verification_statuses"] = [config.verification_status_filter]
    if config.target_modules is not None:
        filters["target_modules"] = list(config.target_modules)
    return filters


def trainer_config_from_finetune_config(config: FinetuneConfig) -> TrainerConfig:
    """Map a first-class FT/SFT config into ``TinyTrainer`` settings."""

    config = FinetuneConfig(**config.to_dict())
    spec = get_training_mode_spec(config.mode)
    lesson_filters = build_finetune_lesson_filter(config)
    policy_mode = spec.expected_policy_mode
    model_type = config.model_type
    target_modules = list(config.target_modules or [])
    trainable_target_modules = None
    use_fast_adapters = bool(config.use_fast_adapters)
    fast_adapter_names = list(config.fast_adapter_names or [])
    train_router = False
    active_adapters = None
    adapter_from_target_modules = config.adapter_from_target_modules
    use_generated_params = bool(config.use_generated_params)
    generated_condition_names = list(config.generated_condition_names or [])
    active_conditions = None
    conditions_from_target_modules = True

    if config.mode == "sft_full":
        policy_mode = "all"
    elif config.mode == "sft_module":
        model_type = "mop_oracle"
        policy_mode = "target_modules_only"
        trainable_target_modules = target_modules
    elif config.mode == "sft_adapter":
        model_type = "mop_oracle" if model_type == "dense" else model_type
        policy_mode = "fast_adapters_only"
        use_fast_adapters = True
        if not fast_adapter_names:
            fast_adapter_names = (
                adapter_names_from_target_modules(target_modules) or ["default"]
            )
        if not target_modules:
            adapter_from_target_modules = False
            active_adapters = list(fast_adapter_names)
    elif config.mode == "sft_generated":
        model_type = "mop_oracle" if model_type == "dense" else model_type
        policy_mode = "generated_params_only"
        use_generated_params = True
        if not generated_condition_names:
            generated_condition_names = (
                condition_names_from_target_modules(target_modules) or ["default"]
            )
        if not target_modules:
            conditions_from_target_modules = False
            active_conditions = list(generated_condition_names)
    elif config.mode == "sft_router":
        model_type = "mop_learned_router"
        policy_mode = "router_only"
        train_router = True
    elif config.mode == "repair_sft":
        policy_mode = "all"
    elif config.mode == "continued_pretraining_smoke":
        policy_mode = "all"

    run_name = f"{config.mode}_{model_type}"
    return TrainerConfig(
        run_name=run_name,
        seed=config.seed,
        model_type=model_type,
        model_ref=config.model_ref,
        lesson_path=config.lesson_path,
        index_path=config.index_path,
        dataset_ref=config.dataset_ref,
        dataset_split=config.dataset_split,
        dataset_version_id=config.dataset_version_id,
        feedback_store_path=config.feedback_store_path,
        tokenizer_type=config.tokenizer_type,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        tokenizer_spec_path=config.tokenizer_spec_path,
        curriculum_strategy=config.curriculum_strategy,
        target_modules=lesson_filters.get("target_modules"),
        curriculum_domains=lesson_filters.get("domains"),
        curriculum_skills=lesson_filters.get("skills"),
        curriculum_verification_statuses=lesson_filters.get("verification_statuses"),
        trainable_policy_mode=policy_mode,
        trainable_target_modules=trainable_target_modules,
        train_router=train_router,
        use_fast_adapters=use_fast_adapters,
        fast_adapter_names=fast_adapter_names or None,
        active_adapters=active_adapters,
        adapter_from_target_modules=adapter_from_target_modules,
        use_generated_params=use_generated_params,
        generated_condition_names=generated_condition_names or None,
        generated_rank=config.generated_rank,
        generated_type=config.generated_type,
        active_conditions=active_conditions,
        conditions_from_target_modules=conditions_from_target_modules,
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        eval_interval=1,
        checkpoint_interval=1,
        eval_batches=config.eval_batches,
        max_seq_len=config.max_seq_len,
        learning_rate=config.learning_rate,
        device=config.device,
        precision=config.precision,
        enable_amp=config.enable_amp,
        allow_tf32=config.allow_tf32,
        deterministic=config.deterministic,
        compile_model=config.compile_model,
        require_device_available=config.require_device_available,
        run_registry_root=config.run_registry_root,
        artifact_root=config.artifact_root,
        save_checkpoints=config.save_checkpoints,
        save_full_checkpoints=config.save_full_checkpoints,
        resume_from_checkpoint=config.resume_from_checkpoint,
        checkpoint_every_steps=config.checkpoint_every_steps,
        training_kind="sft",
        source_config_kind="sft",
        source_config=config.to_dict(),
    )


@dataclass(slots=True)
class FinetuneResult:
    """Structured result for one FT/SFT smoke run."""

    mode: str
    run_id: str
    trainer_result: dict[str, Any]
    mode_spec: dict[str, Any]
    artifacts: dict[str, str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "mode": self.mode,
            "run_id": self.run_id,
            "trainer_result": dict(self.trainer_result),
            "mode_spec": dict(self.mode_spec),
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
        }

    def save_json(self, path: str | Path) -> Path:
        """Write this fine-tune result to JSON and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def run_finetune(config: FinetuneConfig) -> FinetuneResult:
    """Run one CPU-smoke FT/SFT mode through ``TinyTrainer``."""

    config = FinetuneConfig(**config.to_dict())
    spec = get_training_mode_spec(config.mode)
    trainer_config = trainer_config_from_finetune_config(config)
    trainer = TinyTrainer(trainer_config)
    trainer_result = trainer.train()
    artifacts = {
        key: str(value)
        for key, value in trainer_result.artifacts.items()
        if isinstance(value, str)
    }
    metrics = {
        **trainer_result.metrics,
        "finetune_mode": config.mode,
        "finetune_objective": spec.objective,
        "finetune_expected_policy_mode": spec.expected_policy_mode,
        "finetune_filters": build_finetune_lesson_filter(config),
        "finetune_config": config.to_dict(),
    }
    result = FinetuneResult(
        mode=config.mode,
        run_id=trainer_result.run_id,
        trainer_result=trainer_result.to_dict(),
        mode_spec=spec.to_dict(),
        artifacts=artifacts,
        metrics=metrics,
    )
    result_path = result.save_json(trainer.run_dir / "finetune_result.json")
    result.artifacts["finetune_result_json"] = str(result_path)
    result.save_json(result_path)
    return result
