"""Restore consolidated MoP-Forge GPU checkpoints for inference or post-training."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.models.architectures import (
    ModelArchitectureConfig,
    build_tiny_model_from_architecture,
)
from mopforge.tokenization import build_tokenizer, tokenizer_spec_from_config


def load_gpu_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
):
    """Return model, architecture, tokenizer, spec, payload, and restore metadata."""

    from mopforge.gpu.checkpointing import load_gpu_checkpoint, restore_gpu_checkpoint
    from mopforge.gpu.config import GPUTrainingConfig

    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        raise ValueError(
            "This operation requires a consolidated .pt checkpoint; consolidate "
            "a distributed sharded checkpoint before loading it outside FSDP."
        )
    payload = load_gpu_checkpoint(checkpoint, map_location="cpu")
    if config_path is None:
        config_data = dict(payload.get("config") or {})
    else:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config_data = dict(raw.get("payload") or raw)
    if not config_data:
        raise ValueError("Checkpoint has no GPU config snapshot; pass config_path.")
    config = GPUTrainingConfig.from_dict(config_data)
    tokenizer_spec = tokenizer_spec_from_config(config)
    tokenizer = build_tokenizer(tokenizer_spec)
    architecture_data = dict(
        (payload.get("model_metadata") or {}).get("architecture") or {}
    )
    architecture = (
        ModelArchitectureConfig.from_dict(architecture_data)
        if architecture_data
        else architecture_from_gpu_config(config)
    )
    model = build_tiny_model_from_architecture(architecture, tokenizer=tokenizer)
    restore_metadata = restore_gpu_checkpoint(
        payload,
        model=model,
        restore_rng=False,
        restore_optimizer=False,
        restore_scheduler=False,
        restore_scaler=False,
        strict_model=payload.get("trainable_model_state") is None,
    )
    return {
        "model": model,
        "architecture": architecture,
        "tokenizer": tokenizer,
        "tokenizer_spec": tokenizer_spec,
        "payload": payload,
        "config": config,
        "restore_metadata": restore_metadata,
    }


def architecture_from_gpu_config(config) -> ModelArchitectureConfig:
    """Build architecture metadata from a GPUTrainingConfig."""

    return ModelArchitectureConfig(
        name=config.name,
        model_type=config.model_type,
        architecture_family=config.architecture_family,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        intermediate_size=config.intermediate_size,
        n_key_value_heads=config.n_key_value_heads,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        tie_word_embeddings=config.tie_word_embeddings,
        module_names=config.module_names or ["core", "coding", "debugging", "repair"],
        always_include_core=config.always_include_core,
        mop_block_type=config.mop_block_type,
        expert_count=config.expert_count,
        active_experts=config.active_experts,
        routing_granularity=config.routing_granularity,
        shared_depth_ratio=config.shared_depth_ratio,
        use_lora_deltas=config.use_lora_deltas,
        lora_tail_only=config.lora_tail_only,
        lora_rank=config.lora_rank,
        lora_target_modules=config.lora_target_modules,
        use_fast_adapters=config.use_fast_adapters,
        fast_adapter_names=config.fast_adapter_names,
        fast_adapter_bottleneck_dim=config.fast_adapter_bottleneck_dim,
        use_generated_params=config.use_generated_params,
        generated_condition_names=config.generated_condition_names,
        generated_condition_dim=config.generated_condition_dim,
        generated_rank=config.generated_rank,
        generated_type=config.generated_type,
        intended_scale="large_gpu",
    )
