"""Named GPU model/job profile helpers."""

from __future__ import annotations

from mopforge.gpu.config import GPUTrainingConfig


def model_profile_100m_dense() -> GPUTrainingConfig:
    return _profile("100m_dense_a100_smoke", "dense", 100_000_000, 768, 12, 12, 1024, plan_only=False)


def model_profile_100m_mop() -> GPUTrainingConfig:
    return _profile("100m_mop_a100_smoke", "mop_oracle", 100_000_000, 768, 12, 12, 1024, plan_only=False)


def model_profile_500m_dense() -> GPUTrainingConfig:
    return _profile("500m_dense_h100_plan", "dense", 500_000_000, 1024, 24, 16, 2048, plan_only=True)


def model_profile_500m_mop() -> GPUTrainingConfig:
    return _profile("500m_mop_h100_plan", "mop_oracle", 500_000_000, 1024, 24, 16, 2048, plan_only=True)


def model_profile_1b_mop() -> GPUTrainingConfig:
    return _profile("1b_mop_h100_bf16", "mop_oracle", 1_000_000_000, 1536, 28, 16, 2048, plan_only=True)


def model_profile_2b_mop() -> GPUTrainingConfig:
    return _profile("2b_mop_a100_plan", "mop_oracle", 2_000_000_000, 2048, 32, 16, 2048, plan_only=True)


def model_profile_7b_mop() -> GPUTrainingConfig:
    return _profile("7b_mop_h100_plan", "mop_oracle", 7_000_000_000, 4096, 32, 32, 4096, plan_only=True)


def _profile(
    name: str,
    model_type: str,
    parameter_count: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    *,
    plan_only: bool,
) -> GPUTrainingConfig:
    return GPUTrainingConfig(
        name=name,
        model_type=model_type,
        max_steps=100,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        eval_every_steps=50,
        save_every_steps=100,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        target_modules=["coding", "debugging", "repair", "math"],
        use_fast_adapters=model_type != "dense",
        fast_adapter_names=["coding", "debugging", "repair", "math"] if model_type != "dense" else None,
        use_generated_params=model_type != "dense",
        generated_condition_names=["coding", "debugging", "repair", "math"] if model_type != "dense" else None,
        device="cuda",
        precision="bf16",
        enable_amp=True,
        allow_tf32=True,
        require_device_available=False,
        activation_checkpointing=True,
        efficient_attention="auto",
        metadata={
            "parameter_count": parameter_count,
            "target_gpu_memory_gb": 80,
            "plan_only": plan_only,
            "profile": name,
        },
    )
