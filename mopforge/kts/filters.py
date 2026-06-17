"""Filtering helpers for Knowledge Training Store lessons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mopforge.kts.schema import KnowledgeLesson


def filter_lessons(
    lessons: Iterable[KnowledgeLesson],
    *,
    domain: str | None = None,
    skill: str | None = None,
    subskill: str | None = None,
    target_modules: list[str] | tuple[str, ...] | set[str] | str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    verification_status: str | None = None,
    verification_type: str | None = None,
    metadata_contains: Mapping[str, Any] | None = None,
    module_match: str = "any",
) -> list[KnowledgeLesson]:
    """Return lessons matching the provided filter criteria."""

    requested_modules = _normalize_target_modules(target_modules)
    if module_match not in {"any", "all"}:
        raise ValueError("module_match must be either 'any' or 'all'.")
    if metadata_contains is not None and not isinstance(metadata_contains, Mapping):
        raise ValueError("metadata_contains must be a mapping when provided.")

    matched: list[KnowledgeLesson] = []
    for lesson in lessons:
        if domain is not None and lesson.domain != domain:
            continue
        if skill is not None and lesson.skill != skill:
            continue
        if subskill is not None and lesson.subskill != subskill:
            continue
        if min_difficulty is not None and lesson.difficulty < min_difficulty:
            continue
        if max_difficulty is not None and lesson.difficulty > max_difficulty:
            continue
        if (
            verification_status is not None
            and lesson.verification.get("status") != verification_status
        ):
            continue
        if (
            verification_type is not None
            and lesson.verification.get("type") != verification_type
        ):
            continue
        if requested_modules is not None and not _matches_modules(
            lesson, requested_modules, module_match
        ):
            continue
        if metadata_contains is not None and not _metadata_contains(
            lesson.metadata, metadata_contains
        ):
            continue
        matched.append(lesson)

    return matched


def _normalize_target_modules(
    target_modules: list[str] | tuple[str, ...] | set[str] | str | None,
) -> set[str] | None:
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        if not target_modules.strip():
            raise ValueError("target_modules must not contain empty strings.")
        return {target_modules}
    normalized = set(target_modules)
    if not normalized or not all(
        isinstance(module, str) and module.strip() for module in normalized
    ):
        raise ValueError("target_modules must contain non-empty strings.")
    return normalized


def _matches_modules(
    lesson: KnowledgeLesson, requested_modules: set[str], module_match: str
) -> bool:
    lesson_modules = set(lesson.target_modules)
    if module_match == "all":
        return requested_modules.issubset(lesson_modules)
    return bool(requested_modules & lesson_modules)


def _metadata_contains(
    metadata: Mapping[str, Any], expected_items: Mapping[str, Any]
) -> bool:
    return all(metadata.get(key) == value for key, value in expected_items.items())
