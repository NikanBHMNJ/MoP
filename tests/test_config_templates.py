from __future__ import annotations

from pathlib import Path

from mopforge.configs import MoPForgeConfig, dry_run_config, validate_config_envelope
from mopforge.gpu import estimate_from_config, validate_gpu_training_config
from mopforge.configs import gpu_training_config_from_envelope


def test_example_config_templates_validate_or_dry_run() -> None:
    paths = sorted(Path("configs/examples").glob("*.json"))
    assert paths
    for path in paths:
        envelope = MoPForgeConfig.load(path)
        messages = validate_config_envelope(envelope)
        assert not [message for message in messages if message.startswith("ERROR:")], path
        summary = dry_run_config(envelope)
        assert summary["kind"] == envelope.kind


def test_gpu_job_configs_validate_and_estimate_without_execution() -> None:
    paths = sorted(Path("configs/jobs").glob("*.json"))
    assert paths
    for path in paths:
        envelope = MoPForgeConfig.load(path)
        assert envelope.kind == "gpu_train", path
        config = gpu_training_config_from_envelope(envelope)
        messages = validate_gpu_training_config(config)
        assert not [message for message in messages if message.startswith("ERROR:")], path
        estimate = estimate_from_config(config)
        assert estimate.parameter_count > 0
        assert estimate.total_memory_gb_estimate > 0
