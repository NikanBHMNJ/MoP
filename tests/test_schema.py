"""Tests for KnowledgeLesson validation and serialization."""

from __future__ import annotations

import pytest

from mopforge.kts import KnowledgeLesson, LessonValidationError


def make_lesson(**overrides: object) -> KnowledgeLesson:
    data = {
        "id": "lesson-001",
        "domain": "coding",
        "skill": "debugging",
        "subskill": "returns",
        "difficulty": 2,
        "target_modules": ["coding", "debugging"],
        "input": "Fix the function.",
        "expected_output": "Use an explicit return.",
        "verification": {"type": "python_tests", "status": "verified"},
        "metadata": {"language": "python"},
    }
    data.update(overrides)
    return KnowledgeLesson(**data)


def test_valid_lesson_creation_and_round_trip() -> None:
    lesson = make_lesson()

    assert lesson.is_verified is True
    assert lesson.to_dict()["id"] == "lesson-001"

    loaded = KnowledgeLesson.from_dict(lesson.to_dict())
    assert loaded == lesson


def test_invalid_difficulty_rejected() -> None:
    with pytest.raises(LessonValidationError, match="difficulty"):
        make_lesson(difficulty=6)


def test_missing_required_fields_rejected() -> None:
    data = make_lesson().to_dict()
    del data["id"]

    with pytest.raises(LessonValidationError, match="Missing required"):
        KnowledgeLesson.from_dict(data)


def test_verification_status_rejected() -> None:
    with pytest.raises(LessonValidationError, match="verification.status"):
        make_lesson(verification={"type": "python_tests", "status": "unknown"})
