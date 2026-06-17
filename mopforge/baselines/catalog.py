"""Baseline catalog."""

from __future__ import annotations

from mopforge.baselines.config import BaselineSpec


_CATALOG = {
    "dense_full": BaselineSpec("dense_full", "dense", "dense", description="Dense full fine-tuning baseline."),
    "dense_head_only": BaselineSpec(
        "dense_head_only",
        "dense",
        "dense",
        trainable_policy_mode="head_only",
        description="Dense model with LM head trainable.",
    ),
    "mop_oracle_full": BaselineSpec("mop_oracle_full", "mop_oracle", "mop_oracle", description="Oracle-routed TinyMoP full training."),
    "mop_module_only": BaselineSpec(
        "mop_module_only",
        "mop_oracle",
        "mop_oracle",
        trainable_policy_mode="target_modules_only",
        routing_mode="oracle",
        description="Oracle MoP module-only training.",
    ),
    "adapter_only": BaselineSpec(
        "adapter_only",
        "adapter",
        "mop_oracle",
        trainable_policy_mode="fast_adapters_only",
        use_fast_adapters=True,
        routing_mode="oracle",
        description="Fast-adapter-only MoP baseline.",
    ),
    "generated_params_only": BaselineSpec(
        "generated_params_only",
        "generated",
        "mop_oracle",
        trainable_policy_mode="generated_params_only",
        use_generated_params=True,
        routing_mode="oracle",
        description="Generated-parameter-only MoP baseline.",
    ),
    "mop_learned_router": BaselineSpec(
        "mop_learned_router",
        "mop_learned_router",
        "mop_learned_router",
        trainable_policy_mode="router_only",
        routing_mode="learned_router",
        description="Tiny MoP learned-router baseline.",
    ),
    "moe_tiny": BaselineSpec(
        "moe_tiny",
        "moe",
        "mop_oracle",
        trainable_policy_mode="target_modules_only",
        routing_mode="oracle",
        description="Tiny MoE shim backed by TinyMoP module experts for CPU smoke comparisons.",
        metadata={"implementation": "moe_tiny_shim"},
    ),
}


def list_baselines() -> list[BaselineSpec]:
    return [_CATALOG[key] for key in sorted(_CATALOG)]


def get_baseline(name: str) -> BaselineSpec:
    if name not in _CATALOG:
        raise ValueError(f"Unknown baseline: {name}")
    return _CATALOG[name]
