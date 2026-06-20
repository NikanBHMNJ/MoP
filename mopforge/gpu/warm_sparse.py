"""Warm sparse GPU sweep config generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mopforge.configs import MoPForgeConfig


DEFAULT_BOTTLENECKS = [64, 128, 256]
DEFAULT_LEARNING_RATES = [3e-4, 1e-3, 2e-3]
DEFAULT_LORA_RANKS = [4, 8, 16]


def write_warm_sparse_sweep_configs(
    *,
    output_dir: str | Path,
    base_checkpoint: str,
    activation_cache_path: str | None = None,
    dataset_ref: str | None = None,
    dataset_split_id: str | None = None,
    bottlenecks: list[int] | None = None,
    learning_rates: list[float] | None = None,
    lora_ranks: list[int] | None = None,
    max_steps: int = 2000,
    seed: int = 42,
) -> list[Path]:
    """Write warm sparse adapter/norm-head/core-frozen sweep configs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bottleneck_values = list(bottlenecks or DEFAULT_BOTTLENECKS)
    lr_values = list(learning_rates or DEFAULT_LEARNING_RATES)
    lora_rank_values = list(lora_ranks or DEFAULT_LORA_RANKS)
    written: list[Path] = []
    for bottleneck in bottleneck_values:
        for learning_rate in lr_values:
            written.append(
                _write_config(
                    output,
                    _profile_payload(
                        profile="warm_adapters",
                        policy_mode="adapters_only",
                        base_checkpoint=base_checkpoint,
                        bottleneck=bottleneck,
                        learning_rate=learning_rate,
                        max_steps=max_steps,
                        seed=seed,
                        dataset_ref=dataset_ref,
                        dataset_split_id=dataset_split_id,
                        train_norm=False,
                        train_lm_head=False,
                    ),
                )
            )
            written.append(
                _write_config(
                    output,
                    _profile_payload(
                        profile="warm_adapters_norm_head",
                        policy_mode="adapters_norm_head",
                        base_checkpoint=base_checkpoint,
                        bottleneck=bottleneck,
                        learning_rate=learning_rate,
                        max_steps=max_steps,
                        seed=seed,
                        dataset_ref=dataset_ref,
                        dataset_split_id=dataset_split_id,
                        train_norm=True,
                        train_lm_head=True,
                    ),
                )
            )
            if activation_cache_path:
                written.append(
                    _write_config(
                        output,
                        _profile_payload(
                            profile="cached_warm_adapters_norm_head",
                            policy_mode="adapters_norm_head",
                            base_checkpoint=base_checkpoint,
                            bottleneck=bottleneck,
                            learning_rate=learning_rate,
                            max_steps=max_steps,
                            seed=seed,
                            dataset_ref=dataset_ref,
                            dataset_split_id=dataset_split_id,
                            train_norm=True,
                            train_lm_head=True,
                            activation_cache_path=activation_cache_path,
                        ),
                    )
                )
    for learning_rate in lr_values:
        written.append(
            _write_config(
                output,
                _profile_payload(
                    profile="core_frozen_quality",
                    policy_mode="core_frozen",
                    base_checkpoint=base_checkpoint,
                    bottleneck=64,
                    learning_rate=learning_rate,
                    max_steps=max_steps,
                    seed=seed,
                    dataset_ref=dataset_ref,
                    dataset_split_id=dataset_split_id,
                    train_norm=True,
                    train_lm_head=False,
                ),
            )
        )
        for lora_rank in lora_rank_values:
            written.append(
                _write_config(
                    output,
                    _profile_payload(
                        profile=f"warm_lora_norm_head_r{lora_rank}",
                        policy_mode="adapters_norm_head",
                        base_checkpoint=base_checkpoint,
                        bottleneck=64,
                        learning_rate=learning_rate,
                        max_steps=max_steps,
                        seed=seed,
                        dataset_ref=dataset_ref,
                        dataset_split_id=dataset_split_id,
                        train_norm=True,
                        train_lm_head=True,
                        use_fast_adapters=False,
                        use_lora_deltas=True,
                        lora_rank=lora_rank,
                    ),
                )
            )
    return written


def _profile_payload(
    *,
    profile: str,
    policy_mode: str,
    base_checkpoint: str,
    bottleneck: int,
    learning_rate: float,
    max_steps: int,
    seed: int,
    dataset_ref: str | None,
    dataset_split_id: str | None,
    train_norm: bool,
    train_lm_head: bool,
    activation_cache_path: str | None = None,
    use_fast_adapters: bool = True,
    use_lora_deltas: bool = False,
    lora_rank: int = 0,
) -> dict[str, Any]:
    lr_slug = _lr_slug(learning_rate)
    cache_slug = "_cached" if activation_cache_path else ""
    name = f"100m_mop_{profile}_b{bottleneck}_lr{lr_slug}{cache_slug}_efficiency"
    metadata = {
        "description": "Generated warm sparse GPU efficiency sweep profile.",
        "profile": name,
        "base_checkpoint": base_checkpoint,
        "fixed_eval_seed": seed,
        "same_token_budget": True,
        "sweep": {
            "bottleneck": bottleneck,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank if use_lora_deltas else None,
        },
        "train_norm": train_norm,
        "train_lm_head": train_lm_head,
    }
    payload: dict[str, Any] = {
        "name": name,
        "model_type": "mop_oracle",
        "device": "auto",
        "precision": "auto",
        "require_device_available": False,
        "allow_tf32": True,
        "enable_amp": True,
        "activation_checkpointing": True,
        "module_names": ["coding", "debugging", "repair"],
        "always_include_core": False,
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "max_seq_len": 1024,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": max_steps,
        "eval_every_steps": 100,
        "eval_batches": 2,
        "save_every_steps": 500,
        "log_every_steps": 25,
        "scheduler": "cosine",
        "warmup_steps": 50,
        "early_stopping_enabled": False,
        "early_stopping_patience_evals": 5,
        "early_stopping_min_delta": 0.0,
        "learning_rate": learning_rate,
        "max_train_examples": 10000,
        "max_eval_examples": 512,
        "target_modules": ["coding", "debugging", "repair"],
        "use_fast_adapters": use_fast_adapters,
        "trainable_policy_mode": policy_mode,
        "resume_from_checkpoint": base_checkpoint,
        "resume_model_only": True,
        "base_checkpoint_path": base_checkpoint,
        "save_trainable_only_checkpoints": True,
        "run_generation_eval": True,
        "generation_eval_examples": 2,
        "generation_max_new_tokens": 32,
        "metadata": metadata,
    }
    if use_fast_adapters:
        payload["fast_adapter_names"] = ["coding", "debugging", "repair"]
        payload["fast_adapter_bottleneck_dim"] = bottleneck
    if use_lora_deltas:
        payload["use_lora_deltas"] = True
        payload["lora_rank"] = lora_rank
        payload["lora_target_modules"] = ["coding", "debugging", "repair"]
    if activation_cache_path:
        payload["activation_cache_path"] = activation_cache_path
    if dataset_ref:
        payload["dataset_ref"] = dataset_ref
    if dataset_split_id:
        payload["dataset_split_id"] = dataset_split_id
    return payload


def _write_config(output_dir: Path, payload: dict[str, Any]) -> Path:
    path = output_dir / f"{payload['name']}.json"
    MoPForgeConfig(kind="gpu_train", payload=payload).save(path)
    return path


def _lr_slug(value: float) -> str:
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")
