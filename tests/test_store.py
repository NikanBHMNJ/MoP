"""Tests for the JSONL LessonStore."""

from __future__ import annotations

import pytest

from mopforge.kts import KnowledgeLesson, LessonStore, LessonStoreError


def make_lesson(lesson_id: str = "lesson-001", **overrides: object) -> KnowledgeLesson:
    data = {
        "id": lesson_id,
        "domain": "coding",
        "skill": "debugging",
        "subskill": "returns",
        "difficulty": 2,
        "target_modules": ["coding", "debugging"],
        "input": f"Input for {lesson_id}",
        "expected_output": f"Output for {lesson_id}",
        "verification": {"type": "python_tests", "status": "verified"},
        "metadata": {"language": "python"},
    }
    data.update(overrides)
    return KnowledgeLesson(**data)


def test_jsonl_write_read_round_trip(tmp_path) -> None:
    store = LessonStore(tmp_path / "lessons.jsonl")
    first = make_lesson("lesson-001")
    second = make_lesson("lesson-002", difficulty=4)

    store.add(first)
    store.add(second)

    loaded = store.load_all()
    assert loaded == [first, second]
    assert store.count() == 2
    assert store.get_by_id("lesson-002") == second
    assert store.get_by_id("missing") is None


def test_duplicate_id_rejected(tmp_path) -> None:
    store = LessonStore(tmp_path / "lessons.jsonl")
    store.add(make_lesson("duplicate"))

    with pytest.raises(LessonStoreError, match="already exists"):
        store.add(make_lesson("duplicate"))


def test_duplicate_id_allowed_when_configured(tmp_path) -> None:
    store = LessonStore(tmp_path / "lessons.jsonl", allow_duplicate_ids=True)
    store.add(make_lesson("duplicate"))
    store.add(make_lesson("duplicate"))

    assert store.count() == 2


def test_invalid_json_reports_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\nnot-json\n", encoding="utf-8")
    store = LessonStore(path)

    with pytest.raises(LessonStoreError, match="line 2"):
        store.load_all()


def test_deterministic_sampling_with_seed(tmp_path) -> None:
    store = LessonStore(tmp_path / "lessons.jsonl")
    store.add_many(make_lesson(f"lesson-{index}") for index in range(5))

    first = [lesson.id for lesson in store.sample(3, seed=123)]
    second = [lesson.id for lesson in store.sample(3, seed=123)]

    assert first == second
    assert len(first) == 3
