"""Deterministic scoring helpers for feedback-aware curriculum planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mopforge.feedback.store import LessonFeedbackStore


def score_lesson(summary: dict[str, Any], *, repair_boost: float = 1.0) -> float:
    """Score a lesson summary for feedback-weighted scheduling.

    Failed lessons are prioritized above passed lessons. Unseen lessons receive
    a small non-zero score so they remain eligible in deterministic curricula.
    """

    attempts = int(summary.get("attempts") or 0)
    failures = int(summary.get("failures") or 0)
    passes = int(summary.get("passes") or 0)
    avg_loss = summary.get("avg_loss")
    if attempts == 0:
        return 0.1

    loss_component = 0.0
    if isinstance(avg_loss, (int, float)):
        loss_component = max(0.0, min(float(avg_loss), 10.0)) * 0.1

    score = failures * 2.0 + attempts * 0.5 + loss_component - passes * 0.25
    if summary.get("last_failure_type") in {"syntax_error", "runtime_error", "assertion_failed"}:
        score += 0.25
    return max(0.0, score) * float(repair_boost)


def rank_lesson_ids_by_feedback(
    lesson_ids: list[str],
    feedback_store: LessonFeedbackStore | str | Path,
    *,
    reverse: bool = True,
    repair_boost: float = 1.0,
) -> list[str]:
    """Rank lesson IDs by feedback score with deterministic ID tie-breaks."""

    store = (
        feedback_store
        if isinstance(feedback_store, LessonFeedbackStore)
        else LessonFeedbackStore(feedback_store)
    )
    summaries = store.summaries_for_lessons(lesson_ids)
    scored = [
        (
            score_lesson(summaries[lesson_id], repair_boost=repair_boost),
            lesson_id,
        )
        for lesson_id in lesson_ids
    ]
    if reverse:
        return [lesson_id for _score, lesson_id in sorted(scored, key=lambda item: (-item[0], item[1]))]
    return [lesson_id for _score, lesson_id in sorted(scored, key=lambda item: (item[0], item[1]))]
