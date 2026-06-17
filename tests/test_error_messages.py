from __future__ import annotations

import pytest

from mopforge.cli.main import main
from mopforge.configs import MoPForgeConfig
from mopforge.gpu import GPUTrainingConfig, validate_gpu_training_config


def test_invalid_config_path_error_is_clean(capsys) -> None:
    assert main(["config", "validate", "missing_config_for_test.json"]) == 1
    output = capsys.readouterr().out
    assert output.startswith("ERROR:")
    assert "Config file does not exist" in output


def test_invalid_json_error_includes_path(tmp_path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["config", "validate", str(path)]) == 1
    output = capsys.readouterr().out
    assert "Invalid JSON config" in output
    assert str(path) in output


def test_invalid_gpu_config_error_is_clean(tmp_path, capsys) -> None:
    path = tmp_path / "bad_gpu.json"
    MoPForgeConfig(
        kind="gpu_train",
        payload={**GPUTrainingConfig(require_device_available=False).to_dict(), "max_steps": 0},
    ).save(path)
    assert main(["gpu", "validate", str(path)]) == 1
    output = capsys.readouterr().out
    assert "validation=invalid" in output
    assert "max_steps must be a positive integer" in output


def test_unavailable_cuda_required_error_is_actionable(tmp_path, capsys) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine.")
    path = tmp_path / "cuda_required.json"
    config = GPUTrainingConfig(
        name="cuda_required",
        device="cuda",
        precision="bf16",
        require_device_available=True,
    )
    MoPForgeConfig(kind="gpu_train", payload=config.to_dict()).save(path)
    assert main(["gpu", "validate", str(path)]) == 1
    output = capsys.readouterr().out
    assert "CUDA requested but CUDA is unavailable" in output
    assert "require_device_available=false" in output


def test_missing_gpu_resume_checkpoint_suggests_list(capsys) -> None:
    assert main(["gpu", "resume", "definitely-not-a-real-run"]) == 1
    output = capsys.readouterr().out
    assert "mopforge gpu list" in output


def test_invalid_dataset_and_model_refs_are_errors() -> None:
    dataset_config = GPUTrainingConfig(dataset_ref="missing-dataset-ref")
    model_config = GPUTrainingConfig(model_ref="missing-model-ref")

    dataset_messages = validate_gpu_training_config(dataset_config)
    model_messages = validate_gpu_training_config(model_config)

    assert any("dataset_ref could not be resolved" in message for message in dataset_messages)
    assert any("model_ref could not be resolved" in message for message in model_messages)
