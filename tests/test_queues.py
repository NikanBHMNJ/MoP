"""Tests for local module-specific training queues."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.curriculum import CurriculumConfig, CurriculumPlan
from mopforge.feedback import LessonFeedbackRecord, LessonFeedbackStore
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.queues import (
    TrainingQueueItem,
    TrainingQueueStore,
    build_module_queue_from_indexed_store,
    build_queue_items_from_curriculum,
    consume_queue_once,
)


def make_queue_item(
    item_id: str,
    *,
    module: str = "debugging",
    lesson_id: str = "lesson-a",
    priority: float = 1.0,
    status: str = "pending",
    metadata: dict | None = None,
) -> TrainingQueueItem:
    return TrainingQueueItem(
        item_id=item_id,
        module=module,
        lesson_id=lesson_id,
        priority=priority,
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        metadata=metadata or {},
    )


def make_lesson(
    lesson_id: str,
    *,
    modules: list[str] | None = None,
    difficulty: int = 2,
    skill: str = "debugging",
) -> KnowledgeLesson:
    return KnowledgeLesson(
        id=lesson_id,
        domain="coding",
        skill=skill,
        subskill="returns",
        difficulty=difficulty,
        target_modules=modules or ["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "python_tests", "status": "verified"},
        metadata={"test_code": "assert add(1, 2) == 3"},
    )


def make_plan(lesson_ids: list[str]) -> CurriculumPlan:
    return CurriculumPlan(
        lesson_ids=lesson_ids,
        strategy="feedback_weighted",
        counts_by_skill={"debugging": len(lesson_ids)},
        counts_by_domain={"coding": len(lesson_ids)},
        counts_by_verification_status={"verified": len(lesson_ids)},
        counts_by_target_module={"debugging": len(lesson_ids)},
        total=len(lesson_ids),
        metadata={"batch_size": 2},
    )


def test_training_queue_item_dict_round_trip() -> None:
    item = make_queue_item(
        "queue-debugging-lesson-a",
        metadata={"source_score": 2.5},
    )

    loaded = TrainingQueueItem.from_dict(item.to_dict())

    assert loaded == item


def test_training_queue_item_invalid_status_raises() -> None:
    with pytest.raises(ValueError, match="status"):
        make_queue_item("bad-status", status="waiting")


def test_queue_store_schema_creation(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")

    assert store.count() == 0


def test_add_and_get_one_item(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    item = make_queue_item("queue-debugging-lesson-a")

    store.add_item(item)
    loaded = store.get(item.item_id)

    assert loaded == item


def test_add_many_items(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")

    inserted = store.add_many(
        [
            make_queue_item("queue-debugging-lesson-a"),
            make_queue_item("queue-coding-lesson-a", module="coding"),
        ]
    )

    assert inserted == 2
    assert store.count() == 2


def test_claim_next_picks_highest_priority_pending_item(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    store.add_many(
        [
            make_queue_item("low", priority=1.0),
            make_queue_item("high", priority=9.0),
        ]
    )

    claimed = store.claim_next()

    assert claimed is not None
    assert claimed.item_id == "high"
    assert claimed.status == "running"
    assert claimed.attempts == 1


def test_claim_next_filters_by_module(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    store.add_many(
        [
            make_queue_item("coding-high", module="coding", priority=9.0),
            make_queue_item("debugging-low", module="debugging", priority=1.0),
        ]
    )

    claimed = store.claim_next(module="debugging")

    assert claimed is not None
    assert claimed.item_id == "debugging-low"
    assert claimed.module == "debugging"


def test_mark_done_failed_and_skipped_update_status(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    store.add_many(
        [
            make_queue_item("done"),
            make_queue_item("failed"),
            make_queue_item("skipped"),
        ]
    )

    store.mark_done("done", run_id="run-1", metadata={"ok": True})
    store.mark_failed("failed", error="boom")
    store.mark_skipped("skipped", reason="dry_run")

    assert store.get("done").status == "done"
    assert store.get("done").run_id == "run-1"
    assert store.get("done").metadata["ok"] is True
    assert store.get("failed").status == "failed"
    assert store.get("failed").metadata["error"] == "boom"
    assert store.get("skipped").status == "skipped"
    assert store.get("skipped").metadata["skip_reason"] == "dry_run"


def test_counts_by_status_and_module_work(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    store.add_many(
        [
            make_queue_item("coding", module="coding"),
            make_queue_item("debugging", module="debugging"),
            make_queue_item("done", module="debugging", status="done"),
        ]
    )

    assert store.counts_by_status() == {"done": 1, "pending": 2}
    assert store.counts_by_module() == {"coding": 1, "debugging": 2}


def test_queue_builder_creates_module_specific_items() -> None:
    lessons = [
        make_lesson("lesson-a", modules=["coding", "debugging"], difficulty=4),
        make_lesson("lesson-b", modules=["core"], difficulty=1),
    ]
    lessons[1].target_modules = []
    plan = make_plan(["lesson-a", "lesson-b"])

    items = build_queue_items_from_curriculum(plan, lessons)

    assert [item.module for item in items] == ["coding", "debugging", "core"]
    assert items[0].priority == 4.0
    assert items[-1].lesson_id == "lesson-b"


def test_queue_builder_filters_requested_modules() -> None:
    lessons = [make_lesson("lesson-a", modules=["coding", "debugging"])]
    plan = make_plan(["lesson-a"])

    items = build_queue_items_from_curriculum(
        plan,
        lessons,
        modules=["debugging"],
    )

    assert len(items) == 1
    assert items[0].module == "debugging"


def test_queue_builder_can_add_explicit_skill_queue_module() -> None:
    lessons = [make_lesson("repair-a", modules=["coding", "debugging"], skill="repair")]
    plan = make_plan(["repair-a"])

    items = build_queue_items_from_curriculum(
        plan,
        lessons,
        modules=["repair"],
    )

    assert len(items) == 1
    assert items[0].module == "repair"


def test_module_queue_builder_from_indexed_store_uses_feedback_priority(tmp_path) -> None:
    indexed_store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    indexed_store.add(make_lesson("lesson-a", difficulty=1))
    indexed_store.add(make_lesson("lesson-b", difficulty=1))
    feedback_store = LessonFeedbackStore(tmp_path / "feedback.sqlite")
    feedback_store.add_feedback(
        LessonFeedbackRecord(
            lesson_id="lesson-b",
            passed=False,
            failure_type="syntax_error",
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    items = build_module_queue_from_indexed_store(
        indexed_store,
        CurriculumConfig(strategy="feedback_weighted"),
        modules=["debugging"],
        feedback_store=feedback_store,
    )

    assert items[0].lesson_id == "lesson-b"
    assert items[0].priority > items[-1].priority


def test_consumer_can_process_one_item(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    store.add_item(make_queue_item("queue-debugging-lesson-a"))

    result = consume_queue_once(store, run_registry_root=str(tmp_path / "runs"))

    assert result["item_id"] == "queue-debugging-lesson-a"
    assert result["status"] == "done"
    assert result["attempts"] == 1
    assert result["run_id"]
    assert store.get("queue-debugging-lesson-a").status == "done"
    assert Path(tmp_path / "runs" / result["run_id"] / "queue_item.json").exists()


def test_queue_export_json(tmp_path) -> None:
    store = TrainingQueueStore(tmp_path / "queue.sqlite")
    store.add_item(make_queue_item("queue-debugging-lesson-a"))

    path = store.export_json(tmp_path / "queue.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["items"][0]["item_id"] == "queue-debugging-lesson-a"
    assert loaded["counts_by_status"] == {"pending": 1}
