"""Tests for the SQLite KTS metadata index."""

from __future__ import annotations

import sqlite3

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, KnowledgeLesson, LessonIndex, LessonStore


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
        "metadata": {"language": "python", "topic": "functions"},
    }
    data.update(overrides)
    return KnowledgeLesson(**data)


def test_sqlite_schema_creation(tmp_path) -> None:
    index_path = tmp_path / "lessons.sqlite"
    LessonIndex(index_path)

    with sqlite3.connect(index_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert {"lessons", "lesson_modules", "lesson_metadata"} <= table_names


def test_indexing_a_knowledge_lesson(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    lesson = make_lesson()

    index.index_lesson(lesson, jsonl_line=1, jsonl_offset=0, source_path="lessons.jsonl")

    rows = index.query(domain="coding")
    assert rows[0]["id"] == lesson.id
    assert rows[0]["jsonl_line"] == 1


def test_rebuild_index_from_jsonl_store(tmp_path) -> None:
    store = LessonStore(tmp_path / "lessons.jsonl")
    lessons = [make_lesson("one"), make_lesson("two", skill="repair")]
    store.add_many(lessons)
    index = LessonIndex(tmp_path / "lessons.sqlite")

    count = index.rebuild_from_store(store)

    assert count == 2
    assert index.count() == 2
    assert index.query_ids(skill="repair") == ["two"]


def test_query_by_domain_and_skill(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("match"))
    index.index_lesson(make_lesson("other", domain="math", skill="arithmetic"))

    assert index.query_ids(domain="coding", skill="debugging") == ["match"]


def test_query_by_target_module(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("debug", target_modules=["coding", "debugging"]))
    index.index_lesson(make_lesson("math", target_modules=["math"]))

    assert index.query_ids(target_modules=["debugging"]) == ["debug"]


def test_query_by_verification_status(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("verified"))
    index.index_lesson(
        make_lesson(
            "target",
            skill="repair",
            verification={"type": "python_tests", "status": "verified_target"},
        )
    )

    assert index.query_ids(verification_status="verified_target") == ["target"]


def test_query_by_difficulty_range(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("easy", difficulty=1))
    index.index_lesson(make_lesson("medium", difficulty=3))
    index.index_lesson(make_lesson("hard", difficulty=5))

    assert index.query_ids(difficulty_min=2, difficulty_max=4) == ["medium"]


def test_query_by_metadata_key_value(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("python", metadata={"language": "python"}))
    index.index_lesson(make_lesson("rust", metadata={"language": "rust"}))

    assert index.query_ids(metadata={"language": "python"}) == ["python"]


def test_count_by_skill(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("debug"))
    index.index_lesson(make_lesson("repair", skill="repair"))

    assert index.count_by("skill") == {"debugging": 1, "repair": 1}


def test_indexed_lesson_store_add_writes_jsonl_and_index(tmp_path) -> None:
    store = IndexedLessonStore(
        tmp_path / "lessons.jsonl",
        tmp_path / "lessons.sqlite",
    )
    lesson = make_lesson()

    store.add(lesson)

    assert store.load_all() == [lesson]
    assert store.query_ids(target_modules=["debugging"]) == [lesson.id]
    assert store.count(domain="coding") == 1
    assert store.get(lesson.id) == lesson


def test_index_export_query_json(tmp_path) -> None:
    index = LessonIndex(tmp_path / "lessons.sqlite")
    index.index_lesson(make_lesson("one"))

    output = index.export_query_json(tmp_path / "query.json", domain="coding")

    assert output.read_text(encoding="utf-8").count("one") == 1


def test_example_like_indexing_on_tiny_store(tmp_path) -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    store = IndexedLessonStore(
        tmp_path / "lessons.jsonl",
        tmp_path / "lessons.sqlite",
    )
    for lesson in lessons:
        store.add(lesson)

    assert store.count() == len(lessons)
    assert store.count(target_modules=["debugging"]) == len(lessons)
    assert "debugging" in store.count_by("target_module")
