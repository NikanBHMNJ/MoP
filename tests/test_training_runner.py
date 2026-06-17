"""Tests for curriculum-driven tiny training runs and registry."""

from __future__ import annotations

import math

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore
from mopforge.runs import RunRegistry, TinyTrainingRunConfig, TrainingRunRecord
from mopforge.training import run_tiny_training_from_curriculum


def build_tiny_indexed_store(tmp_path) -> IndexedLessonStore:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    for lesson in generate_coding_bugfix_lessons(count_per_category=1, verify=False):
        store.add(lesson)
    return store


def tiny_config(tmp_path, *, model_type: str = "dense") -> TinyTrainingRunConfig:
    return TinyTrainingRunConfig(
        run_name=f"{model_type}_test",
        model_type=model_type,
        curriculum_strategy="balanced",
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        batch_size=2,
        train_steps=1,
        eval_batches=1,
        max_seq_len=256,
        d_model=32,
        n_layers=1,
        n_heads=2,
    )


def test_tiny_training_run_config_defaults_are_cpu_safe() -> None:
    config = TinyTrainingRunConfig()

    assert config.batch_size == 2
    assert config.train_steps == 3
    assert config.eval_batches == 2
    assert config.max_seq_len == 512
    assert config.d_model == 64
    assert config.n_layers == 2
    assert config.n_heads == 2
    assert config.run_generation_eval is False


def test_training_run_record_json_round_trip(tmp_path) -> None:
    record = TrainingRunRecord(
        run_id="run-1",
        run_name="demo",
        model_type="dense",
        curriculum_strategy="balanced",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        config={"train_steps": 1},
        metrics={"finite": True},
        artifacts={"metrics_json": "metrics.json"},
    )

    path = record.save_json(tmp_path / "run.json")
    loaded = TrainingRunRecord.load_json(path)

    assert loaded == record


def test_run_registry_saves_lists_and_loads_records(tmp_path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    record = TrainingRunRecord(
        run_id="run-1",
        run_name="demo",
        model_type="dense",
        curriculum_strategy="balanced",
        started_at="start",
        finished_at="finish",
        config={},
        metrics={"finite": True},
        artifacts={},
    )

    registry.save(record)

    assert registry.list_runs() == ["run-1"]
    loaded = registry.load("run-1")
    assert loaded.run_id == "run-1"
    assert (tmp_path / "runs" / "run-1" / "metrics.json").exists()


def test_curriculum_runner_supports_dense_model(tmp_path) -> None:
    build_tiny_indexed_store(tmp_path)
    registry = RunRegistry(tmp_path / "runs")

    record = run_tiny_training_from_curriculum(
        tiny_config(tmp_path, model_type="dense"),
        registry=registry,
    )

    assert record.model_type == "dense"
    assert record.metrics["finite"] is True
    assert math.isfinite(record.metrics["train_loss_last"])
    assert registry.load(record.run_id).run_id == record.run_id


def test_curriculum_runner_supports_oracle_mop_model(tmp_path) -> None:
    build_tiny_indexed_store(tmp_path)

    record = run_tiny_training_from_curriculum(
        tiny_config(tmp_path, model_type="mop_oracle"),
        registry=RunRegistry(tmp_path / "runs"),
    )

    assert record.model_type == "mop_oracle"
    assert record.metrics["finite"] is True
    assert record.metrics["curriculum_total"] > 0


def test_curriculum_runner_metrics_have_required_keys(tmp_path) -> None:
    build_tiny_indexed_store(tmp_path)

    record = run_tiny_training_from_curriculum(
        tiny_config(tmp_path, model_type="dense"),
        registry=RunRegistry(tmp_path / "runs"),
    )

    required = {
        "train_loss_last",
        "eval_loss_mean",
        "finite",
        "train_examples",
        "eval_examples",
        "curriculum_total",
        "counts_by_skill",
        "counts_by_verification_status",
    }
    assert required <= set(record.metrics)
    assert "run_json" in record.artifacts
    assert "metrics_json" in record.artifacts
    assert "curriculum_plan_json" in record.artifacts


def test_curriculum_runner_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
