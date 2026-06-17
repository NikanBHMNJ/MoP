"""Collect repair failures and write repair lessons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from mopforge.kts import KnowledgeLesson, LessonStore
from mopforge.repair.lesson_builder import build_repair_lesson_from_failure
from mopforge.repair.schema import RepairFailureRecord


def failure_record_from_generation_result(
    result: dict[str, Any],
    source_lesson: KnowledgeLesson,
) -> RepairFailureRecord | None:
    """Create a failure record from a failed generation result."""

    if result.get("passed") is True:
        return None

    source_lesson.validate()
    failure_type = str(result.get("failure_type") or "unknown_failure")
    return RepairFailureRecord(
        lesson_id=str(result.get("lesson_id") or source_lesson.id),
        original_input=source_lesson.input,
        expected_output=source_lesson.expected_output,
        generated_text=str(result.get("generated_text", "")),
        candidate_code=str(result.get("candidate_code", "")),
        failure_type=failure_type,
        verifier_stdout=str(result.get("stdout", result.get("verifier_stdout", ""))),
        verifier_stderr=str(result.get("stderr", result.get("verifier_stderr", ""))),
        exit_code=result.get("exit_code"),
        timeout=bool(result.get("timeout", False)),
        target_modules=list(result.get("target_modules") or source_lesson.target_modules),
        metadata={
            "source_difficulty": source_lesson.difficulty,
            "source_domain": source_lesson.domain,
            "source_skill": source_lesson.skill,
            "source_subskill": source_lesson.subskill,
            "generation_result": dict(result),
        },
    )


def build_repair_lessons_from_generation_results(
    results: Iterable[dict[str, Any]],
    lessons_by_id: dict[str, KnowledgeLesson],
) -> list[KnowledgeLesson]:
    """Build repair lessons for all failed generation results."""

    repair_lessons: list[KnowledgeLesson] = []
    for result in results:
        lesson_id = result.get("lesson_id")
        if lesson_id not in lessons_by_id:
            continue
        failure = failure_record_from_generation_result(
            result, lessons_by_id[str(lesson_id)]
        )
        if failure is not None:
            repair_lessons.append(build_repair_lesson_from_failure(failure))
    return repair_lessons


def write_repair_lessons(
    repair_lessons: list[KnowledgeLesson],
    output_path: str | Path,
    *,
    allow_duplicates: bool = False,
) -> int:
    """Write repair lessons to a JSONL store and return the count written."""

    path = Path(output_path)
    if path.exists() and not allow_duplicates:
        path.unlink()
    store = LessonStore(path, allow_duplicate_ids=allow_duplicates)
    store.add_many(repair_lessons)
    return len(repair_lessons)
