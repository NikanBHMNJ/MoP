"""Tests for the coding bug-fix lesson builder."""

from __future__ import annotations

from mopforge.builders import BUG_CATEGORIES, generate_coding_bugfix_lessons
from mopforge.kts import LessonStore
from mopforge.verify import verify_python_solution


def test_generated_lessons_validate_correctly() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=1)

    assert len(lessons) == len(BUG_CATEGORIES)
    for lesson in lessons:
        lesson.validate()
        assert lesson.domain == "coding"
        assert lesson.skill == "debugging"
        assert "coding" in lesson.target_modules
        assert "debugging" in lesson.target_modules
        assert lesson.is_verified is True


def test_generated_examples_have_required_metadata() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=2)

    for lesson in lessons:
        metadata = lesson.metadata
        assert metadata["language"] == "python"
        assert isinstance(metadata["function_name"], str)
        assert metadata["bug_type"] in BUG_CATEGORIES
        assert isinstance(metadata["test_names"], list)
        assert metadata["test_names"]
        assert isinstance(metadata["test_code"], str)
        assert metadata["test_code"].strip()


def test_generated_tests_reject_buggy_inputs() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=1)

    for lesson in lessons:
        result = verify_python_solution(lesson.input, lesson.metadata["test_code"])
        assert result.passed is False


def test_generated_demo_set_has_no_duplicate_ids() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=False)
    ids = [lesson.id for lesson in lessons]

    assert len(lessons) == 50
    assert len(ids) == len(set(ids))


def test_lesson_store_can_load_and_filter_generated_lessons(tmp_path) -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=2)
    store = LessonStore(tmp_path / "coding_bugfix_lessons.jsonl")
    store.add_many(lessons)

    loaded = store.load_all()
    filtered = store.filter(
        domain="coding",
        skill="debugging",
        target_modules=["coding", "debugging"],
        module_match="all",
        verification_status="verified",
    )

    assert loaded == lessons
    assert len(filtered) == len(lessons)
