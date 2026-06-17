"""Builders for local module-specific training queue items."""

from __future__ import annotations

from dataclasses import replace
import re

from mopforge.curriculum import CurriculumConfig, CurriculumPlan, CurriculumScheduler
from mopforge.feedback import LessonFeedbackStore, score_lesson
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.queues.schema import TrainingQueueItem


DETERMINISTIC_QUEUE_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def build_queue_items_from_curriculum(
    plan: CurriculumPlan,
    lessons: list[KnowledgeLesson],
    *,
    modules: list[str] | None = None,
    source: str = "curriculum",
    priority_by_feedback: dict[str, float] | None = None,
) -> list[TrainingQueueItem]:
    """Create module-specific queue items from a curriculum plan."""

    requested_modules = set(modules) if modules is not None else None
    lesson_by_id = {lesson.id: lesson for lesson in lessons}
    items: list[TrainingQueueItem] = []
    for lesson_id in plan.lesson_ids:
        lesson = lesson_by_id.get(lesson_id)
        if lesson is None:
            continue
        lesson_modules = _lesson_modules(lesson)
        if (
            requested_modules is not None
            and lesson.skill in requested_modules
            and lesson.skill not in lesson_modules
        ):
            lesson_modules.append(lesson.skill)
        if requested_modules is not None:
            lesson_modules = [
                module for module in lesson_modules if module in requested_modules
            ]
        priority = (
            float(priority_by_feedback[lesson.id])
            if priority_by_feedback is not None and lesson.id in priority_by_feedback
            else float(lesson.difficulty)
        )
        for module in lesson_modules:
            items.append(
                TrainingQueueItem(
                    item_id=f"queue-{_slug(module)}-{_slug(lesson.id)}",
                    module=module,
                    lesson_id=lesson.id,
                    priority=priority,
                    source=source,
                    created_at=DETERMINISTIC_QUEUE_TIMESTAMP,
                    updated_at=DETERMINISTIC_QUEUE_TIMESTAMP,
                    metadata={
                        "curriculum_strategy": plan.strategy,
                        "lesson_difficulty": lesson.difficulty,
                    },
                )
            )
    return items


def build_module_queue_from_indexed_store(
    indexed_store: IndexedLessonStore,
    config: CurriculumConfig,
    modules: list[str] | None = None,
    feedback_store: LessonFeedbackStore | None = None,
) -> list[TrainingQueueItem]:
    """Build queue items from an indexed store and curriculum config."""

    scheduler = CurriculumScheduler(indexed_store=indexed_store)
    plan_config = config
    if feedback_store is not None and config.feedback_store_path is None:
        plan_config = replace(config, feedback_store_path=str(feedback_store.path))
    plan = scheduler.build_plan(plan_config)
    lessons = scheduler.load_lessons(plan)
    priority_by_feedback = None
    if feedback_store is not None:
        summaries = feedback_store.summaries_for_lessons(plan.lesson_ids)
        priority_by_feedback = {
            lesson_id: score_lesson(summary, repair_boost=config.repair_boost)
            for lesson_id, summary in summaries.items()
        }
    return build_queue_items_from_curriculum(
        plan,
        lessons,
        modules=modules,
        source=config.strategy,
        priority_by_feedback=priority_by_feedback,
    )


def _lesson_modules(lesson: KnowledgeLesson) -> list[str]:
    modules = list(lesson.target_modules) if lesson.target_modules else ["core"]
    seen: set[str] = set()
    ordered = []
    for module in modules:
        if module not in seen:
            ordered.append(module)
            seen.add(module)
    return ordered


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "item"
