"""JSONL-backed storage for Knowledge Training Store lessons."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from mopforge.kts.exceptions import LessonStoreError, LessonValidationError
from mopforge.kts.filters import filter_lessons
from mopforge.kts.schema import KnowledgeLesson


class LessonStore:
    """A small JSONL-backed lesson store.

    Each lesson is stored as one JSON object per line. This v0.1 store is
    designed for local research workflows and simple training pipelines, not
    concurrent multi-writer database use.
    """

    def __init__(self, path: str | Path, *, allow_duplicate_ids: bool = False) -> None:
        """Create a store bound to ``path``.

        Args:
            path: JSONL file path.
            allow_duplicate_ids: If False, adding a lesson with an existing ID
                raises ``LessonStoreError``.
        """

        self.path = Path(path)
        self.allow_duplicate_ids = allow_duplicate_ids
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, lesson: KnowledgeLesson) -> None:
        """Validate and append one lesson to the JSONL store."""

        lesson.validate()
        if not self.allow_duplicate_ids and self.get_by_id(lesson.id) is not None:
            raise LessonStoreError(f"Lesson with id {lesson.id!r} already exists.")

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    def add_many(self, lessons: Iterable[KnowledgeLesson]) -> None:
        """Validate and append many lessons to the JSONL store."""

        lesson_list = list(lessons)
        for lesson in lesson_list:
            lesson.validate()

        if not self.allow_duplicate_ids:
            existing_ids = {lesson.id for lesson in self.iter_lessons()}
            seen_ids: set[str] = set()
            for lesson in lesson_list:
                if lesson.id in existing_ids or lesson.id in seen_ids:
                    raise LessonStoreError(
                        f"Lesson with id {lesson.id!r} already exists."
                    )
                seen_ids.add(lesson.id)

        with self.path.open("a", encoding="utf-8") as file:
            for lesson in lesson_list:
                file.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> list[KnowledgeLesson]:
        """Load and validate every lesson in the store."""

        return list(self.iter_lessons())

    def iter_lessons(self) -> Iterator[KnowledgeLesson]:
        """Iterate over lessons in the store, skipping empty lines."""

        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise LessonStoreError(
                        f"Invalid JSON in {self.path} at line {line_number}: "
                        f"{exc.msg}."
                    ) from exc

                try:
                    yield KnowledgeLesson.from_dict(data)
                except LessonValidationError as exc:
                    raise LessonStoreError(
                        f"Invalid lesson in {self.path} at line {line_number}: {exc}"
                    ) from exc

    def count(self) -> int:
        """Return the number of lessons in the store."""

        return sum(1 for _ in self.iter_lessons())

    def get_by_id(self, lesson_id: str) -> KnowledgeLesson | None:
        """Return the first lesson with ``lesson_id``, or None if absent."""

        for lesson in self.iter_lessons():
            if lesson.id == lesson_id:
                return lesson
        return None

    def filter(self, **filters: object) -> list[KnowledgeLesson]:
        """Load and return lessons matching KTS filter arguments."""

        return filter_lessons(self.iter_lessons(), **filters)

    def sample(
        self, n: int, seed: int | None = None, **filters: object
    ) -> list[KnowledgeLesson]:
        """Sample up to ``n`` lessons, optionally after applying filters.

        Sampling is without replacement. Providing ``seed`` makes the returned
        sample deterministic for a fixed store and filter set.
        """

        if type(n) is not int or n < 0:
            raise LessonStoreError("n must be a non-negative integer.")

        lessons = self.filter(**filters) if filters else self.load_all()
        if n == 0:
            return []
        if n >= len(lessons):
            return list(lessons)

        rng = random.Random(seed)
        return rng.sample(lessons, n)
