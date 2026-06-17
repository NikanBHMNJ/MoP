"""Tests for the feedback-weighted retraining loop."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.feedback import LessonFeedbackRecord, LessonFeedbackStore
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.loops import (
    FeedbackRetrainingConfig,
    FeedbackRetrainingResult,
    run_feedback_retraining_loop,
    summarize_feedback_delta,
)


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


def build_tiny_loop_store(tmp_path) -> tuple[IndexedLessonStore, LessonFeedbackStore]:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))

    feedback_store = LessonFeedbackStore(tmp_path / "feedback.sqlite")
    feedback_store.add_feedback(
        LessonFeedbackRecord(
            lesson_id="lesson-b",
            passed=False,
            failure_type="syntax_error",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )
    return store, feedback_store


def tiny_loop_config(tmp_path) -> FeedbackRetrainingConfig:
    return FeedbackRetrainingConfig(
        loop_name="loop_test",
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        feedback_store_path=str(tmp_path / "feedback.sqlite"),
        run_registry_root=str(tmp_path / "runs"),
        batch_size=1,
        train_steps=1,
        eval_batches=1,
        generation_eval_examples=1,
        max_new_tokens=4,
        max_seq_len=128,
        d_model=16,
        n_layers=1,
        n_heads=2,
    )


def test_feedback_retraining_config_defaults_are_cpu_safe() -> None:
    config = FeedbackRetrainingConfig()

    assert config.model_type == "dense"
    assert config.curriculum_strategy == "feedback_weighted"
    assert config.batch_size == 2
    assert config.train_steps == 3
    assert config.eval_batches == 1
    assert config.generation_eval_examples == 2
    assert config.max_seq_len == 512
    assert config.d_model == 64


def test_feedback_retraining_result_json_round_trip(tmp_path) -> None:
    result = FeedbackRetrainingResult(
        loop_id="loop-1",
        loop_name="demo",
        model_type="dense",
        curriculum_strategy="feedback_weighted",
        train_run_id="run-1",
        feedback_records_added=1,
        eval_examples=1,
        pass_count=0,
        fail_count=1,
        failures_by_type={"syntax_error": 1},
        artifacts={"loop_result_json": "loop_result.json"},
        metrics={"finite": True},
    )

    path = result.save_json(tmp_path / "loop_result.json")
    loaded = FeedbackRetrainingResult.load_json(path)

    assert loaded == result


def test_feedback_retraining_loop_runs_and_adds_feedback(tmp_path) -> None:
    _store, feedback_store = build_tiny_loop_store(tmp_path)
    before_count = feedback_store.count()

    result = run_feedback_retraining_loop(tiny_loop_config(tmp_path))

    after_store = LessonFeedbackStore(tmp_path / "feedback.sqlite")
    assert result.train_run_id is not None
    assert result.feedback_records_added == 1
    assert after_store.count() == before_count + 1
    assert result.eval_examples == 1
    assert result.pass_count + result.fail_count == 1
    assert isinstance(result.failures_by_type, dict)


def test_feedback_retraining_loop_writes_artifacts(tmp_path) -> None:
    build_tiny_loop_store(tmp_path)

    result = run_feedback_retraining_loop(tiny_loop_config(tmp_path))

    required = {
        "loop_result_json",
        "generation_eval_after_retraining_json",
        "feedback_export_after_retraining_json",
        "train_run_json",
        "train_metrics_json",
        "train_curriculum_plan_json",
    }
    assert required <= set(result.artifacts)
    for path in result.artifacts.values():
        assert path
        assert Path(path).exists()

    loaded = json.loads(
        Path(result.artifacts["generation_eval_after_retraining_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(loaded) == 1


def test_summarize_feedback_delta() -> None:
    delta = summarize_feedback_delta(
        {"feedback_count": 2, "failure_counts": {"syntax_error": 1}},
        {"feedback_count": 4, "failure_counts": {"syntax_error": 2}},
    )

    assert delta["feedback_count_before"] == 2
    assert delta["feedback_count_after"] == 4
    assert delta["new_feedback_records"] == 2


def test_feedback_retraining_loop_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
