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


def test_explicit_optimizer_budget_runs_exact_accumulated_updates(tmp_path):
    result = GPUTrainer(
        _config(
            tmp_path,
            max_steps=999,
            max_optimizer_steps=3,
            gradient_accumulation_steps=2,
            eval_every_optimizer_steps=3,
            save_every_optimizer_steps=3,
            log_every_optimizer_steps=1,
        )
    ).train()

    assert result.metrics["global_steps"] == 6
    assert result.metrics["optimizer_steps"] == 3
    assert result.metrics["microstep_budget"] == 6
    assert result.metrics["optimizer_step_budget"] == 3
    assert result.metrics["budget_source"] == "max_optimizer_steps"


def test_trainer_records_real_epoch_boundaries(tmp_path):
    result = GPUTrainer(_config(tmp_path, max_steps=6, save_every_steps=6)).train()

    assert result.metrics["train_epoch"] >= 2
    assert result.metrics["shuffle_train"] is True
    assert result.metrics["train_shuffle_seed"] == 42


def test_full_eval_consumes_entire_eval_loader(tmp_path):
    trainer = GPUTrainer(_config(tmp_path, eval_full_dataset=True, eval_batches=1))
    trainer.setup()

    metrics = trainer.evaluate()

    assert metrics["eval_examples"] == trainer.data_metadata["eval_examples"]
    assert metrics["eval_batches"] == len(trainer.eval_loader)
    assert metrics["eval_full_dataset"] is True


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


def test_model_only_warm_start_loads_before_distributed_wrapping(tmp_path, monkeypatch):
    first = GPUTrainer(_config(tmp_path, max_steps=1)).train()
    checkpoint = first.artifacts["latest_checkpoint_path"]
    trainer = GPUTrainer(
        _config(
            tmp_path,
            resume_from_checkpoint=checkpoint,
            resume_model_only=True,
        )
    )
    import mopforge.gpu.trainer as trainer_module

    events = []
    original_wrap = trainer_module.wrap_distributed_model
    original_load = trainer.load_checkpoint

    def tracked_wrap(*args, **kwargs):
        events.append("wrap")
        return original_wrap(*args, **kwargs)

    def tracked_load(*args, **kwargs):
        events.append("load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "wrap_distributed_model", tracked_wrap)
    monkeypatch.setattr(trainer, "load_checkpoint", tracked_load)
    trainer.setup()
    trainer.close()

    assert events[:2] == ["load", "wrap"]


def test_token_budget_supersedes_legacy_microstep_limit(tmp_path):
    result = GPUTrainer(
        _config(
            tmp_path,
            max_steps=1,
            max_train_tokens=150,
            scheduler="cosine",
            scheduler_unit="tokens",
            warmup_tokens=50,
            save_every_steps=100,
            eval_every_steps=100,
        )
    ).train()

    assert result.metrics["tokens_seen"] >= 150
    assert result.metrics["global_steps"] > 1
    assert result.metrics["budget_source"] == "max_train_tokens"


def test_resume_restores_exact_epoch_batch_cursor(tmp_path):
    first = GPUTrainer(_config(tmp_path, max_steps=2, save_every_steps=2)).train()
    resumed = GPUTrainer(
        _config(
            tmp_path,
            max_steps=3,
            resume_from_checkpoint=first.artifacts["latest_checkpoint_path"],
        )
    )
    resumed.setup()

    assert resumed.data_metadata["resume_cursor"]["exact_within_epoch"] is True
    assert resumed.data_metadata["resume_cursor"]["batches_skipped"] == 2


def test_checkpoint_payload_contains_gpu_metadata(tmp_path):
    result = GPUTrainer(_config(tmp_path)).train()
    import torch

    payload = torch.load(result.artifacts["latest_checkpoint_path"], map_location="cpu", weights_only=False)
    assert payload["metadata"]["training_kind"] == "gpu_train"
    assert payload["metadata"]["tokens_seen"] >= 0
    assert payload["runtime_metadata"]["selected_device"] == "cpu"
    assert "scaler_state" in payload
