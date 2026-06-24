import json

import pytest

from mopforge.gpu import GPUTrainingConfig, validate_gpu_training_config
from mopforge.runtime import build_runtime_context


def test_gpu_training_config_roundtrip_and_save(tmp_path):
    config = GPUTrainingConfig(
        name="unit_gpu",
        max_steps=1,
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_seq_len=64,
        require_device_available=False,
    )
    loaded = GPUTrainingConfig.from_dict(config.to_dict())
    assert loaded.name == "unit_gpu"
    path = config.save(tmp_path / "gpu.json")
    assert GPUTrainingConfig.load(path).max_steps == 1
    json.dumps(config.to_dict())


def test_gpu_training_config_validation_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        GPUTrainingConfig(max_steps=0)
    with pytest.raises(ValueError):
        GPUTrainingConfig(d_model=17, n_heads=2)


def test_optimizer_update_budget_and_legacy_microstep_budget_are_explicit():
    legacy = GPUTrainingConfig(max_steps=5, gradient_accumulation_steps=2)
    assert legacy.microstep_budget == 5
    assert legacy.optimizer_step_budget == 3

    explicit = GPUTrainingConfig(
        max_steps=999,
        max_optimizer_steps=4,
        gradient_accumulation_steps=3,
        warmup_optimizer_steps=2,
    )
    assert explicit.microstep_budget == 12
    assert explicit.optimizer_step_budget == 4
    assert explicit.scheduler_warmup_optimizer_steps == 2


def test_gpu_training_config_runtime_cpu_and_auto_work_on_cpu_only():
    cpu = GPUTrainingConfig(device="cpu", precision="fp32")
    assert build_runtime_context(cpu_runtime(cpu)).device_info.selected == "cpu"
    auto = GPUTrainingConfig(device="auto", precision="auto", require_device_available=False)
    assert build_runtime_context(cpu_runtime(auto)).device_info.selected in {"cpu", "cuda:0", "mps"}


def test_gpu_training_config_fp8_is_planning_safe():
    config = GPUTrainingConfig(precision="fp8", require_device_available=False)
    runtime = build_runtime_context(cpu_runtime(config))
    assert runtime.precision_policy.fp8_requested is True
    assert runtime.precision_policy.selected in {"fp32", "fp16", "bf16"}


def test_gpu_config_validation_warns_for_missing_token_artifacts(tmp_path):
    config = GPUTrainingConfig(
        token_shard_manifest=str(tmp_path / "missing-manifest.json"),
        tokenizer_type="hf",
        tokenizer_spec_path=str(tmp_path / "missing-tokenizer-spec.json"),
        require_device_available=False,
    )

    messages = validate_gpu_training_config(config)

    assert any("token_shard_manifest does not exist" in message for message in messages)
    assert any("tokenizer_spec_path does not exist" in message for message in messages)


def test_distributed_training_rejects_primary_only_generation_eval():
    with pytest.raises(ValueError, match="Distributed generated-code evaluation"):
        GPUTrainingConfig(
            architecture_family="production_decoder_v2",
            distributed_strategy="fsdp",
            distributed_checkpoint_mode="sharded",
            run_generation_eval=True,
        )


def cpu_runtime(config):
    from mopforge.runtime import RuntimeConfig

    return RuntimeConfig(
        device=config.device,
        precision=config.precision,
        enable_amp=config.enable_amp,
        allow_tf32=config.allow_tf32,
        require_device_available=config.require_device_available,
    )
