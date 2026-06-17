import json
from pathlib import Path

import pytest

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.gpu import GPUTrainer, GPUTrainingConfig
from mopforge.kts import LessonStore


pytest.importorskip("torch")


def _lessons(tmp_path: Path) -> str:
    path = tmp_path / "lessons.jsonl"
    if path.exists():
        return str(path)
    store = LessonStore(path)
    store.add_many(
        lesson for lesson in generate_coding_bugfix_lessons(count_per_category=1) if lesson.is_verified
    )
    return str(path)


def _config(tmp_path: Path, **overrides) -> GPUTrainingConfig:
    payload = {
        "name": "test_gpu",
        "lesson_path": _lessons(tmp_path),
        "output_root": str(tmp_path / "gpu_runs"),
        "artifact_root": str(tmp_path / "artifacts"),
        "max_steps": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "eval_every_steps": 1,
        "eval_batches": 1,
        "save_every_steps": 1,
        "log_every_steps": 1,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "max_seq_len": 64,
        "device": "auto",
        "precision": "auto",
        "require_device_available": False,
        "max_train_examples": 4,
        "max_eval_examples": 2,
    }
    payload.update(overrides)
    return GPUTrainingConfig(**payload)


def test_gpu_trainer_one_step_cpu_fallback_writes_outputs(tmp_path):
    result = GPUTrainer(_config(tmp_path)).train()
    assert result.status == "completed"
    assert result.metrics["runtime"]["selected_device"] == "cpu"
    assert result.metrics["optimizer_steps"] == 1
    assert Path(result.artifacts["gpu_training_result_json"]).exists()
    assert Path(result.artifacts["latest_checkpoint_path"]).exists()
    assert Path(result.output_dir, "metrics.json").exists()


def test_gradient_accumulation_changes_optimizer_step_count(tmp_path):
    result = GPUTrainer(
        _config(tmp_path, max_steps=4, gradient_accumulation_steps=2, save_every_steps=4)
    ).train()
    assert result.metrics["global_steps"] == 4
    assert result.metrics["optimizer_steps"] == 2
    assert result.metrics["effective_batch_size"] == 2


def test_amp_scaler_disabled_on_cpu_and_metadata_recorded(tmp_path):
    result = GPUTrainer(_config(tmp_path, precision="fp16", enable_amp=True)).train()
    assert result.metrics["scaler"]["enabled"] is False
    assert result.metrics["runtime"]["selected_precision"] == "fp32"


def test_checkpoint_resume_continues_global_step(tmp_path):
    first = GPUTrainer(_config(tmp_path, max_steps=1)).train()
    checkpoint = first.artifacts["latest_checkpoint_path"]
    resumed = GPUTrainer(_config(tmp_path, max_steps=2, resume_from_checkpoint=checkpoint)).train()
    assert resumed.metrics["global_steps"] == 2
    assert resumed.metrics["optimizer_steps"] >= first.metrics["optimizer_steps"]


def test_checkpoint_payload_contains_gpu_metadata(tmp_path):
    result = GPUTrainer(_config(tmp_path)).train()
    import torch

    payload = torch.load(result.artifacts["latest_checkpoint_path"], map_location="cpu", weights_only=False)
    assert payload["metadata"]["training_kind"] == "gpu_train"
    assert payload["metadata"]["tokens_seen"] >= 0
    assert payload["runtime_metadata"]["selected_device"] == "cpu"
    assert "scaler_state" in payload
