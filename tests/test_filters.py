"""Tests for KTS filtering helpers."""

from __future__ import annotations

from mopforge.kts import KnowledgeLesson, LessonStore
from mopforge.kts.filters import filter_lessons


def make_lesson(lesson_id: str, **overrides: object) -> KnowledgeLesson:
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
        "metadata": {"language": "python", "source": "unit"},
    }
    data.update(overrides)
    return KnowledgeLesson(**data)


def test_filter_by_domain_skill_module_and_status() -> None:
    lessons = [
        make_lesson("match"),
        make_lesson("wrong-domain", domain="math", target_modules=["math"]),
        make_lesson("partial", verification={"type": "python_tests", "status": "partial"}),
        make_lesson("missing-debugging-module", target_modules=["coding"]),
    ]

    filtered = filter_lessons(
        lessons,
        domain="coding",
        skill="debugging",
        target_modules=["coding", "debugging"],
        module_match="all",
        verification_status="verified",
    )

    assert [lesson.id for lesson in filtered] == ["match"]


def test_filter_by_difficulty_verification_type_and_metadata(tmp_path) -> None:
    store = LessonStore(tmp_path / "lessons.jsonl")
    store.add_many(
        [
            make_lesson("easy", difficulty=1, metadata={"language": "python"}),
            make_lesson(
                "medium",
                difficulty=3,
                verification={"type": "human_review", "status": "verified"},
                metadata={"language": "python", "source": "docs"},
            ),
            make_lesson("hard", difficulty=5, metadata={"language": "rust"}),
        ]
    )

    filtered = store.filter(
        min_difficulty=2,
        max_difficulty=4,
        verification_type="human_review",
        metadata_contains={"source": "docs"},
    )

    assert [lesson.id for lesson in filtered] == ["medium"]


def test_filter_target_modules_any_match() -> None:
    lessons = [
        make_lesson("coding", target_modules=["coding"]),
        make_lesson("math", domain="math", target_modules=["math"]),
    ]

    filtered = filter_lessons(lessons, target_modules=["debugging", "coding"])

    assert [lesson.id for lesson in filtered] == ["coding"]
