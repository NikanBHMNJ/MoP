"""JSONL lesson store with an additive SQLite metadata index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mopforge.kts.index import LessonIndex
from mopforge.kts.schema import KnowledgeLesson
from mopforge.kts.store import LessonStore


class IndexedLessonStore:
    """Combine canonical JSONL storage with a SQLite metadata index."""

    def __init__(
        self,
        lesson_path: str | Path,
        index_path: str | Path | None = None,
        *,
        auto_rebuild: bool = False,
        allow_duplicate_ids: bool = False,
    ) -> None:
        """Create an indexed lesson store.

        Args:
            lesson_path: Canonical JSONL lesson path.
            index_path: SQLite index path. Defaults to ``lesson_path`` with a
                ``.sqlite`` suffix.
            auto_rebuild: If True, rebuild the index from JSONL on startup.
            allow_duplicate_ids: Passed through to the underlying LessonStore.
        """

        self.store = LessonStore(lesson_path, allow_duplicate_ids=allow_duplicate_ids)
        self.lesson_path = self.store.path
        self.index_path = Path(index_path) if index_path is not None else self.lesson_path.with_suffix(".sqlite")
        self.index = LessonIndex(self.index_path)
        if auto_rebuild:
            self.rebuild_index()

    def add(self, lesson: KnowledgeLesson) -> None:
        """Append a lesson to JSONL and index it."""

        next_line = self.store.count() + 1
        self.store.add(lesson)
        self.index.index_lesson(
            lesson,
            jsonl_line=next_line,
            source_path=self.lesson_path,
        )

    def load_all(self) -> list[KnowledgeLesson]:
        """Load all lessons from canonical JSONL."""

        return self.store.load_all()

    def query_ids(self, **filters: Any) -> list[str]:
        """Return matching lesson IDs from the SQLite index."""

        return self.index.query_ids(**filters)

    def query(self, **filters: Any) -> list[dict[str, Any]]:
        """Return matching indexed metadata rows."""

        return self.index.query(**filters)

    def filter(self, **filters: Any) -> list[KnowledgeLesson]:
        """Return full lessons matching index filters."""

        ids = set(self.query_ids(**filters))
        return [lesson for lesson in self.store.iter_lessons() if lesson.id in ids]

    def get(self, lesson_id: str) -> KnowledgeLesson | None:
        """Return one full lesson by ID from JSONL."""

        return self.store.get_by_id(lesson_id)

    def count(self, **filters: Any) -> int:
        """Return count of indexed lessons matching filters."""

        return self.index.count(**filters)

    def count_by(self, field: str, **filters: Any) -> dict[str | None, int]:
        """Return grouped counts from the SQLite index."""

        return self.index.count_by(field, **filters)

    def rebuild_index(self) -> int:
        """Rebuild the SQLite index from canonical JSONL."""

        return self.index.rebuild_from_store(self.store)
