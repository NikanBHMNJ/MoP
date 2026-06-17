"""Tests for the CPU-first TinyTrainer skeleton."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.training import TinyTrainer, TrainerConfig, TrainerResult, TrainerState


def make_lesson(lesson_id: str) -> KnowledgeLesson:
    return KnowledgeLesson(
        id=lesson_id,
        domain="coding",
        skill="debugging",
        subskill="missing-return",
        difficulty=1,
        target_modules=["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "python_tests", "status": "verified"},
        metadata={"test_code": "assert add(1, 2) == 3"},
    )


def build_tiny_store(tmp_path) -> IndexedLessonStore:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))
    return store


def tiny_config(
    tmp_path,
    *,
    model_type: str = "dense",
    max_steps: int = 1,
    resume_from=None,
    **overrides,
) -> TrainerConfig:
    values = dict(
        run_name=f"{model_type}_trainer_test",
        model_type=model_type,
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        batch_size=1,
        max_steps=max_steps,
        eval_interval=1,
        checkpoint_interval=1,
        eval_batches=1,
        max_seq_len=128,
        d_model=16,
        n_layers=1,
        n_heads=2,
        run_registry_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        resume_from=resume_from,
    )
    values.update(overrides)
    return TrainerConfig(**values)


def test_trainer_config_defaults_are_cpu_safe() -> None:
    config = TrainerConfig()

    assert config.device == "cpu"
    assert config.use_amp is False
    assert config.batch_size == 2
    assert config.max_steps == 3
    assert config.save_checkpoints is True
    assert config.use_fast_adapters is False


def test_trainer_state_dict_round_trip() -> None:
    state = TrainerState(
        global_step=2,
        best_eval_loss=1.5,
        checkpoint_artifacts=["ckpt-1"],
        metrics_history=[{"step": 1, "loss": 2.0}],
    )

    loaded = TrainerState.from_dict(state.to_dict())

    assert loaded == state


def test_trainer_result_json_save(tmp_path) -> None:
    result = TrainerResult(
        run_id="run-1",
        run_name="demo",
        model_type="dense",
        routing_mode="none",
        final_state={"global_step": 1},
        metrics={"finite": True},
        artifacts={"metrics_json": "metrics.json"},
        finite=True,
    )

    path = result.save_json(tmp_path / "trainer_result.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["run_id"] == "run-1"
    assert loaded["finite"] is True


def test_tiny_trainer_setup_works_on_tiny_kts(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    trainer = TinyTrainer(tiny_config(tmp_path))

    trainer.setup()

    assert trainer.model is not None
    assert trainer.plan.total == 2
    assert trainer.train_lessons
    assert trainer.eval_lessons


def test_dense_trainer_runs_one_step_and_returns_finite_metrics(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(tiny_config(tmp_path, model_type="dense")).train()

    assert result.model_type == "dense"
    assert result.finite is True
    assert result.final_state["global_step"] == 1
    assert math.isfinite(result.metrics["train_loss_last"])
    assert math.isfinite(result.metrics["eval_loss_mean"])


def test_oracle_mop_trainer_runs_one_step_and_returns_finite_metrics(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(tiny_config(tmp_path, model_type="mop_oracle")).train()

    assert result.model_type == "mop_oracle"
    assert result.routing_mode == "oracle"
    assert result.finite is True
    assert result.metrics["checkpoint_count"] >= 1


def test_oracle_mop_trainer_runs_one_step_with_target_module_policy(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(
        tiny_config(
            tmp_path,
            model_type="mop_oracle",
            trainable_policy_mode="target_modules_only",
            trainable_target_modules=["coding"],
        )
    ).train()

    group_summaries = {
        summary["name"]: summary
        for summary in result.metrics["parameter_group_summaries"]
    }

    assert result.finite is True
    assert result.metrics["trainable_policy"]["mode"] == "target_modules_only"
    assert result.metrics["parameter_counts"]["trainable"] > 0
    assert result.metrics["parameter_counts"]["frozen"] > 0
    assert group_summaries["module:coding"]["trainable_params"] > 0
    assert group_summaries["module:debugging"]["trainable_params"] == 0


def test_learned_router_mop_trainer_runs_one_step(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(tiny_config(tmp_path, model_type="mop_learned_router")).train()

    assert result.model_type == "mop_learned_router"
    assert result.routing_mode == "learned_router"
    assert result.finite is True


def test_trainer_result_includes_parameter_counts_and_summaries(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(tiny_config(tmp_path, model_type="dense")).train()

    assert result.metrics["parameter_counts"]["total"] > 0
    assert result.metrics["parameter_counts"]["trainable"] > 0
    assert result.metrics["parameter_group_summaries"]
    assert result.final_state["parameter_counts"] == result.metrics["parameter_counts"]
    assert result.final_state["parameter_group_summaries"]


def test_trainer_runs_one_step_with_fast_adapters_only_policy(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(
        tiny_config(
            tmp_path,
            model_type="mop_oracle",
            use_fast_adapters=True,
            fast_adapter_names=["coding", "debugging", "repair"],
            fast_adapter_bottleneck_dim=4,
            trainable_policy_mode="fast_adapters_only",
        )
    ).train()
    group_summaries = {
        summary["name"]: summary
        for summary in result.metrics["parameter_group_summaries"]
    }

    assert result.finite is True
    assert result.metrics["trainable_policy"]["mode"] == "fast_adapters_only"
    assert result.metrics["adapter_metadata"]["enabled"] is True
    assert result.metrics["adapter_metadata"]["adapter_names"] == [
        "coding",
        "debugging",
        "repair",
    ]
    assert group_summaries["adapter:coding"]["trainable_params"] > 0
    assert group_summaries["shared_core"]["trainable_params"] == 0
    assert result.metrics["parameter_counts"]["trainable"] > 0
    assert result.metrics["parameter_counts"]["frozen"] > 0


def test_checkpoint_save_registers_artifact(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    trainer = TinyTrainer(tiny_config(tmp_path))
    trainer.setup()
    trainer.state.global_step = 1

    checkpoint = trainer.save_checkpoint(step=1)

    assert checkpoint is not None
    assert Path(checkpoint.path).exists()
    assert trainer.artifact_manager.get(checkpoint.artifact_id) == checkpoint


def test_checkpoint_load_into_fresh_trainer_works(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    trainer = TinyTrainer(tiny_config(tmp_path))
    trainer.setup()
    checkpoint = trainer.save_checkpoint(step=1)

    fresh = TinyTrainer(tiny_config(tmp_path))
    fresh.setup()
    fresh.load_checkpoint(checkpoint.path)

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, fresh.model.state_dict()[key])


def test_resume_from_checkpoint_runs_without_crashing(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    first = TinyTrainer(tiny_config(tmp_path, max_steps=1))
    first_result = first.train()
    checkpoint_path = first_result.artifacts["checkpoint_paths"][-1]

    resumed = TinyTrainer(tiny_config(tmp_path, max_steps=2, resume_from=checkpoint_path))
    result = resumed.train()

    assert result.final_state["global_step"] == 2
    assert result.finite is True


def test_artifact_manifest_gets_checkpoint_metrics_and_result_entries(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    trainer = TinyTrainer(tiny_config(tmp_path))
    result = trainer.train()

    kinds = [record.kind for record in trainer.artifact_manager.list(run_id=result.run_id)]

    assert "checkpoint" in kinds
    assert "metrics" in kinds
    assert "config" in kinds
    assert Path(result.artifacts["trainer_result_json"]).exists()


def test_tiny_trainer_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
