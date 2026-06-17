import json

import pytest

from mopforge.gpu import GPUTrainingConfig
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


def cpu_runtime(config):
    from mopforge.runtime import RuntimeConfig

    return RuntimeConfig(
        device=config.device,
        precision=config.precision,
        enable_amp=config.enable_amp,
        allow_tf32=config.allow_tf32,
        require_device_available=config.require_device_available,
    )
