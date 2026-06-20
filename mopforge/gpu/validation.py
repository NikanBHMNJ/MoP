"""Validation helpers for GPU job configs."""

from __future__ import annotations

from pathlib import Path

from mopforge.gpu.config import GPUTrainingConfig
from mopforge.gpu.memory import estimate_from_config
from mopforge.datasets import DatasetRegistry
from mopforge.models import ModelRegistry
from mopforge.runtime import RuntimeConfig, build_runtime_context


def validate_gpu_training_config(config: GPUTrainingConfig) -> list[str]:
    messages: list[str] = []
    try:
        GPUTrainingConfig.from_dict(config.to_dict())
    except Exception as exc:
        messages.append(f"ERROR: {exc}")
        return messages
    if config.metadata.get("plan_only") and not config.metadata.get("allow_train_plan"):
        messages.append("WARNING: this job profile is a planning/validation profile.")
    if config.lesson_path and not Path(config.lesson_path).exists() and not config.dataset_ref and not config.corpus_path:
        messages.append(f"WARNING: lesson_path does not exist yet: {config.lesson_path}")
    if config.corpus_path and not Path(config.corpus_path).exists():
        messages.append(f"WARNING: corpus_path does not exist yet: {config.corpus_path}")
    if config.activation_cache_path and not Path(config.activation_cache_path).exists():
        messages.append(f"WARNING: activation_cache_path does not exist yet: {config.activation_cache_path}")
    if config.dataset_ref:
        try:
            DatasetRegistry().resolve_dataset_ref(config.dataset_ref)
        except Exception as exc:
            messages.append(
                "ERROR: dataset_ref could not be resolved: "
                f"{config.dataset_ref}. {exc}"
            )
    if config.model_ref:
        try:
            ModelRegistry().resolve_model_ref(config.model_ref)
        except Exception as exc:
            messages.append(
                "ERROR: model_ref could not be resolved: "
                f"{config.model_ref}. {exc}"
            )
    try:
        build_runtime_context(
            RuntimeConfig(
                device=config.device,
                precision=config.precision,
                enable_amp=config.enable_amp,
                allow_tf32=config.allow_tf32,
                deterministic=config.deterministic,
                compile_model=config.compile_model,
                require_device_available=config.require_device_available,
            )
        )
    except Exception as exc:
        messages.append(
            "ERROR: runtime device/precision request cannot execute locally: "
            f"{exc}. Use device='auto' or set require_device_available=false for planning."
        )
    estimate = estimate_from_config(config)
    for warning in estimate.warnings:
        messages.append(f"WARNING: {warning}")
    if estimate.fits is False:
        messages.append("WARNING: memory estimator says this profile may not fit the target GPU.")
    return messages


def dry_run_gpu_training_config(config: GPUTrainingConfig) -> dict:
    estimate = estimate_from_config(config)
    return {
        "kind": "gpu_train",
        "name": config.name,
        "model_type": config.model_type,
        "device": config.device,
        "precision": config.precision,
        "max_steps": config.max_steps,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.effective_batch_size,
        "activation_checkpointing": config.activation_checkpointing,
        "efficient_attention": config.efficient_attention,
        "module_names": list(config.module_names or []),
        "always_include_core": config.always_include_core,
        "mop_block_type": config.mop_block_type,
        "resume_model_only": config.resume_model_only,
        "save_trainable_only_checkpoints": config.save_trainable_only_checkpoints,
        "activation_cache_path": config.activation_cache_path,
        "dataset_split_id": config.dataset_split_id,
        "run_generation_eval": config.run_generation_eval,
        "early_stopping_enabled": config.early_stopping_enabled,
        "early_stopping_patience_evals": config.early_stopping_patience_evals,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "output_root": config.output_root,
        "memory_estimate": estimate.to_dict(),
        "warnings": validate_gpu_training_config(config),
    }
