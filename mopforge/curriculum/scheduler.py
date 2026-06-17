"""Deterministic curriculum planning over indexed KTS lessons."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopforge.curriculum.config import CurriculumConfig
from mopforge.kts import IndexedLessonStore, KnowledgeLesson, LessonIndex


@dataclass(slots=True)
class CurriculumPlan:
    """Result of a curriculum scheduling pass."""

    lesson_ids: list[str]
    strategy: str
    counts_by_skill: dict[str, int]
    counts_by_domain: dict[str, int]
    counts_by_verification_status: dict[str, int]
    counts_by_target_module: dict[str, int]
    total: int
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable curriculum plan."""

        return {
            "lesson_ids": list(self.lesson_ids),
            "strategy": self.strategy,
            "counts_by_skill": dict(self.counts_by_skill),
            "counts_by_domain": dict(self.counts_by_domain),
            "counts_by_verification_status": dict(
                self.counts_by_verification_status
            ),
            "counts_by_target_module": dict(self.counts_by_target_module),
            "total": self.total,
            "metadata": dict(self.metadata or {}),
        }

    def save_json(self, path: str | Path) -> Path:
        """Write the plan to JSON and return the output path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return output_path


class CurriculumScheduler:
    """Build deterministic lesson plans from a KTS SQLite index."""

    SUPPORTED_STRATEGIES = {
        "sequential",
        "shuffled",
        "balanced",
        "module_targeted",
        "repair_boosted",
        "feedback_weighted",
    }

    def __init__(
        self,
        indexed_store: IndexedLessonStore | None = None,
        index: LessonIndex | None = None,
    ) -> None:
        """Create a scheduler from an indexed store or standalone index."""

        if indexed_store is None and index is None:
            raise ValueError("indexed_store or index is required.")
        self.indexed_store = indexed_store
        self.index = index if index is not None else indexed_store.index  # type: ignore[union-attr]

    def build_plan(self, config: CurriculumConfig) -> CurriculumPlan:
        """Build a deterministic curriculum plan for ``config``."""

        if config.strategy not in self.SUPPORTED_STRATEGIES:
            valid = ", ".join(sorted(self.SUPPORTED_STRATEGIES))
            raise ValueError(f"Unsupported strategy. Valid strategies: {valid}.")

        rows = self._eligible_rows(config)
        lesson_ids = self._order_ids(rows, config)
        if config.max_lessons is not None:
            lesson_ids = lesson_ids[: config.max_lessons]
        return self._plan_from_ids(lesson_ids, config, rows)

    def load_lessons(self, plan: CurriculumPlan) -> list[KnowledgeLesson]:
        """Load full ``KnowledgeLesson`` objects for a plan."""

        if self.indexed_store is None:
            raise ValueError("load_lessons requires an IndexedLessonStore.")
        lessons: list[KnowledgeLesson] = []
        for lesson_id in plan.lesson_ids:
            lesson = self.indexed_store.get(lesson_id)
            if lesson is not None:
                lessons.append(lesson)
        return lessons

    def iter_batches(
        self, plan: CurriculumPlan, batch_size: int | None = None
    ) -> Iterator[list[str]]:
        """Yield plan lesson IDs in deterministic batches."""

        size = batch_size or int((plan.metadata or {}).get("batch_size", 1))
        if size <= 0:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(plan.lesson_ids), size):
            yield plan.lesson_ids[start : start + size]

    def _eligible_rows(self, config: CurriculumConfig) -> list[dict[str, Any]]:
        query_kwargs: dict[str, Any] = {
            "difficulty_min": config.difficulty_min,
            "difficulty_max": config.difficulty_max,
        }
        if config.target_modules:
            query_kwargs["target_modules"] = config.target_modules
            query_kwargs["module_match"] = "any"

        rows = self.index.query(
            **{key: value for key, value in query_kwargs.items() if value is not None}
        )
        filtered = []
        for row in rows:
            if config.domains is not None and row["domain"] not in config.domains:
                continue
            if config.skills is not None and row["skill"] not in config.skills:
                continue
            if (
                config.verification_statuses is not None
                and row["verification_status"] not in config.verification_statuses
            ):
                continue
            if not config.include_repair and _is_repair_like(row):
                continue
            filtered.append(row)
        return filtered

    def _order_ids(
        self, rows: list[dict[str, Any]], config: CurriculumConfig
    ) -> list[str]:
        if config.strategy == "sequential":
            return sorted(str(row["id"]) for row in rows)
        if config.strategy == "shuffled":
            ids = sorted(str(row["id"]) for row in rows)
            random.Random(config.seed).shuffle(ids)
            return ids
        if config.strategy == "module_targeted":
            return sorted(str(row["id"]) for row in rows)
        if config.strategy == "repair_boosted":
            repair_ids = sorted(str(row["id"]) for row in rows if _is_repair_like(row))
            other_ids = sorted(str(row["id"]) for row in rows if not _is_repair_like(row))
            return repair_ids + other_ids
        if config.strategy == "feedback_weighted":
            return _feedback_weighted_ids(rows, config)
        return _balanced_ids(rows)

    def _plan_from_ids(
        self,
        lesson_ids: list[str],
        config: CurriculumConfig,
        rows: list[dict[str, Any]],
    ) -> CurriculumPlan:
        row_by_id = {str(row["id"]): row for row in rows}
        selected_rows = [row_by_id[lesson_id] for lesson_id in lesson_ids if lesson_id in row_by_id]
        modules_by_id = self.index.modules_by_ids(lesson_ids)
        target_module_counts: Counter[str] = Counter()
        for lesson_id in lesson_ids:
            target_module_counts.update(modules_by_id.get(lesson_id, []))
        return CurriculumPlan(
            lesson_ids=list(lesson_ids),
            strategy=config.strategy,
            counts_by_skill=dict(Counter(str(row["skill"]) for row in selected_rows)),
            counts_by_domain=dict(Counter(str(row["domain"]) for row in selected_rows)),
            counts_by_verification_status=dict(
                Counter(str(row["verification_status"]) for row in selected_rows)
            ),
            counts_by_target_module=dict(target_module_counts),
            total=len(lesson_ids),
            metadata={
                "batch_size": config.batch_size,
                "seed": config.seed,
                "repair_boost": config.repair_boost,
                "feedback_store_path": config.feedback_store_path,
                "duplicate_ids_used": False,
            },
        )


def _is_repair_like(row: dict[str, Any]) -> bool:
    return row.get("skill") == "repair" or row.get("verification_status") == "verified_target"


def _balanced_ids(rows: list[dict[str, Any]]) -> list[str]:
    by_skill: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_skill[str(row["skill"])].append(str(row["id"]))
    for ids in by_skill.values():
        ids.sort()

    ordered: list[str] = []
    skill_names = sorted(by_skill)
    while any(by_skill.values()):
        for skill in skill_names:
            if by_skill[skill]:
                ordered.append(by_skill[skill].pop(0))
    return ordered


def _feedback_weighted_ids(
    rows: list[dict[str, Any]],
    config: CurriculumConfig,
) -> list[str]:
    lesson_ids = sorted(str(row["id"]) for row in rows)
    if not config.feedback_store_path:
        return lesson_ids

    feedback_path = Path(config.feedback_store_path)
    if not feedback_path.exists():
        return lesson_ids

    from mopforge.feedback import rank_lesson_ids_by_feedback

    return rank_lesson_ids_by_feedback(
        lesson_ids,
        feedback_path,
        repair_boost=config.repair_boost,
    )
