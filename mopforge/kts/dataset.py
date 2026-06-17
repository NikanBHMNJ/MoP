"""Dataset wrappers for Knowledge Training Store lessons."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mopforge.kts.schema import KnowledgeLesson


class LessonDataset:
    """A lightweight, framework-agnostic dataset over knowledge lessons."""

    def __init__(self, lessons: list[KnowledgeLesson]) -> None:
        """Create a dataset from a list of validated lessons."""

        self.lessons = list(lessons)
        for lesson in self.lessons:
            lesson.validate()

    def __len__(self) -> int:
        """Return the number of lessons."""

        return len(self.lessons)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one lesson as a plain dictionary suitable for ML loops."""

        lesson = self.lessons[idx]
        return lesson_to_dataset_item(lesson)


def lesson_to_dataset_item(lesson: KnowledgeLesson) -> dict[str, Any]:
    """Convert a lesson to the stable dataset item format."""

    return {
        "id": lesson.id,
        "input": lesson.input,
        "expected_output": lesson.expected_output,
        "domain": lesson.domain,
        "skill": lesson.skill,
        "subskill": lesson.subskill,
        "target_modules": list(lesson.target_modules),
        "difficulty": lesson.difficulty,
        "verification": deepcopy(lesson.verification),
        "metadata": deepcopy(lesson.metadata),
    }


try:
    import torch
except ImportError:
    torch = None
    TorchLessonDataset = None
else:

    class TorchLessonDataset(torch.utils.data.Dataset):  # type: ignore[name-defined]
        """Optional PyTorch Dataset wrapper for KTS lessons."""

        def __init__(self, lessons: list[KnowledgeLesson]) -> None:
            self._dataset = LessonDataset(lessons)

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self._dataset[idx]
