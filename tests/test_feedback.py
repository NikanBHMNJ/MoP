"""Tests for lesson feedback records, store, scoring, and curriculum integration."""

from __future__ import annotations

import json

from mopforge.curriculum import CurriculumConfig, CurriculumScheduler
from mopforge.feedback import (
    LessonFeedbackRecord,
    LessonFeedbackStore,
    feedback_records_from_generation_eval,
    rank_lesson_ids_by_feedback,
    score_lesson,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson


def make_lesson(lesson_id: str, skill: str = "debugging") -> KnowledgeLesson:
    return KnowledgeLesson(
        id=lesson_id,
        domain="coding",
        skill=skill,
        subskill="returns",
        difficulty=2,
        target_modules=["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "unit_tests", "status": "verified"},
        metadata={"test_code": "assert add(1, 2) == 3"},
    )


def build_indexed_store(tmp_path) -> IndexedLessonStore:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    for lesson_id in ["lesson-a", "lesson-b", "lesson-c"]:
        store.add(make_lesson(lesson_id))
    return store


def test_feedback_record_dict_round_trip() -> None:
    record = LessonFeedbackRecord(
        lesson_id="lesson-a",
        run_id="run-1",
        model_type="tiny_dense",
        curriculum_strategy="balanced",
        passed=False,
        failure_type="syntax_error",
        loss=1.25,
        generated=True,
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"exit_code": 1},
    )

    loaded = LessonFeedbackRecord.from_dict(record.to_dict())

    assert loaded.to_dict() == record.to_dict()


def test_feedback_store_schema_creation(tmp_path) -> None:
    store = LessonFeedbackStore(tmp_path / "feedback.sqlite")

    assert store.count() == 0


def test_add_one_feedback_record_updates_count(tmp_path) -> None:
    store = LessonFeedbackStore(tmp_path / "feedback.sqlite")

    row_id = store.add_feedback(
        LessonFeedbackRecord(
            lesson_id="lesson-a",
            passed=True,
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    assert row_id == 1
    assert store.count() == 1


def test_failed_and_passed_records_update_summary(tmp_path) -> None:
    store = LessonFeedbackStore(tmp_path / "feedback.sqlite")

    store.add_many(
        [
            LessonFeedbackRecord(
                lesson_id="lesson-a",
                passed=False,
                failure_type="syntax_error",
                loss=4.0,
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            LessonFeedbackRecord(
                lesson_id="lesson-a",
                passed=True,
                loss=2.0,
                timestamp="2026-01-01T00:01:00+00:00",
            ),
            LessonFeedbackRecord(
                lesson_id="lesson-a",
                passed=None,
                timestamp="2026-01-01T00:02:00+00:00",
            ),
        ]
    )

    summary = store.summary_for_lesson("lesson-a")

    assert summary["attempts"] == 3
    assert summary["passes"] == 1
    assert summary["failures"] == 1
    assert summary["avg_loss"] == 3.0
    assert summary["last_failure_type"] == "syntax_error"


def test_failure_counts_by_type(tmp_path) -> None:
    store = LessonFeedbackStore(tmp_path / "feedback.sqlite")

    store.add_many(
        [
            LessonFeedbackRecord(lesson_id="a", passed=False, failure_type="syntax_error"),
            LessonFeedbackRecord(lesson_id="b", passed=False, failure_type="syntax_error"),
            LessonFeedbackRecord(lesson_id="c", passed=False, failure_type="timeout"),
            LessonFeedbackRecord(lesson_id="d", passed=True, failure_type=None),
        ]
    )

    assert store.failure_counts_by_type() == {"syntax_error": 2, "timeout": 1}


def test_import_from_generated_eval_results() -> None:
    records = feedback_records_from_generation_eval(
        [
            {
                "model": "tiny_dense",
                "routing": "none",
                "results": [
                    {
                        "lesson_id": "lesson-a",
                        "passed": False,
                        "failure_type": "syntax_error",
                        "exit_code": 1,
                        "timeout": False,
                    }
                ],
            }
        ],
        run_id="run-1",
        curriculum_strategy="balanced",
    )

    assert len(records) == 1
    assert records[0].lesson_id == "lesson-a"
    assert records[0].model_type == "tiny_dense"
    assert records[0].generated is True
    assert records[0].metadata["exit_code"] == 1


def test_scorer_ranks_failed_lessons_above_passed_lessons(tmp_path) -> None:
    store = LessonFeedbackStore(tmp_path / "feedback.sqlite")
    store.add_many(
        [
            LessonFeedbackRecord(lesson_id="passed", passed=True),
            LessonFeedbackRecord(lesson_id="failed", passed=False, failure_type="syntax_error"),
        ]
    )

    assert score_lesson(store.summary_for_lesson("failed")) > score_lesson(
        store.summary_for_lesson("passed")
    )
    assert rank_lesson_ids_by_feedback(["passed", "failed"], store) == [
        "failed",
        "passed",
    ]


def test_feedback_weighted_curriculum_strategy_is_deterministic(tmp_path) -> None:
    indexed_store = build_indexed_store(tmp_path)
    feedback_store = LessonFeedbackStore(tmp_path / "feedback.sqlite")
    feedback_store.add_many(
        [
            LessonFeedbackRecord(lesson_id="lesson-b", passed=False, failure_type="syntax_error"),
            LessonFeedbackRecord(lesson_id="lesson-a", passed=True),
        ]
    )
    scheduler = CurriculumScheduler(indexed_store=indexed_store)
    config = CurriculumConfig(
        strategy="feedback_weighted",
        feedback_store_path=str(feedback_store.path),
    )

    first = scheduler.build_plan(config)
    second = scheduler.build_plan(config)

    assert first.lesson_ids == second.lesson_ids
    assert first.lesson_ids[0] == "lesson-b"


def test_feedback_weighted_fallback_when_store_missing(tmp_path) -> None:
    scheduler = CurriculumScheduler(indexed_store=build_indexed_store(tmp_path))

    plan = scheduler.build_plan(
        CurriculumConfig(
            strategy="feedback_weighted",
            feedback_store_path=str(tmp_path / "missing.sqlite"),
        )
    )

    assert plan.lesson_ids == sorted(plan.lesson_ids)


def test_feedback_store_export_json(tmp_path) -> None:
    store = LessonFeedbackStore(tmp_path / "feedback.sqlite")
    store.add_feedback(LessonFeedbackRecord(lesson_id="lesson-a", passed=False))

    path = store.export_json(tmp_path / "feedback.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["feedback"][0]["lesson_id"] == "lesson-a"
    assert loaded["summaries"][0]["failures"] == 1
