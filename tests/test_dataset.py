"""Tests for KTS dataset wrappers."""

from __future__ import annotations

from mopforge.kts import KnowledgeLesson, LessonDataset
from mopforge.kts.dataset import TorchLessonDataset


def make_lesson() -> KnowledgeLesson:
    return KnowledgeLesson(
        id="lesson-001",
        domain="coding",
        skill="debugging",
        subskill="returns",
        difficulty=2,
        target_modules=["coding", "debugging"],
        input="Fix the function.",
        expected_output="Use an explicit return.",
        verification={"type": "python_tests", "status": "verified"},
        metadata={"language": "python"},
    )


def test_dataset_item_format() -> None:
    dataset = LessonDataset([make_lesson()])

    item = dataset[0]

    assert len(dataset) == 1
    assert item == {
        "id": "lesson-001",
        "input": "Fix the function.",
        "expected_output": "Use an explicit return.",
        "domain": "coding",
        "skill": "debugging",
        "subskill": "returns",
        "target_modules": ["coding", "debugging"],
        "difficulty": 2,
        "verification": {"type": "python_tests", "status": "verified"},
        "metadata": {"language": "python"},
    }


def test_optional_torch_dataset_behavior() -> None:
    if TorchLessonDataset is None:
        assert TorchLessonDataset is None
        return

    dataset = TorchLessonDataset([make_lesson()])

    assert len(dataset) == 1
    assert dataset[0]["id"] == "lesson-001"
