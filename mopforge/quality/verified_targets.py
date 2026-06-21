"""Helpers for small-model verified-output quality framing."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from mopforge.formatting import FIXED_CODE_XML_FORMAT
from mopforge.kts import KnowledgeLesson


VERIFIED_TARGET_STATUSES = {"verified", "verified_target"}


def frame_verified_target_lesson(
    lesson: KnowledgeLesson,
    *,
    teacher_source: str = "known_verified_target",
    output_format: str = FIXED_CODE_XML_FORMAT,
    require_verified: bool = True,
) -> KnowledgeLesson:
    """Return a lesson clone with verified-output metadata for quality training."""

    lesson.validate()
    status = str(lesson.verification.get("status") or "")
    if require_verified and status not in VERIFIED_TARGET_STATUSES:
        raise ValueError(
            "verified target framing requires verification.status in "
            f"{sorted(VERIFIED_TARGET_STATUSES)}, got {status!r} for {lesson.id}."
        )
    is_verified_target = status in VERIFIED_TARGET_STATUSES
    metadata = {
        **dict(lesson.metadata),
        "quality_output_format": output_format,
        "verified_teacher_target": is_verified_target,
        "teacher_source": teacher_source,
        "quality_track": "small_specialist_verified_code",
    }
    verification = dict(lesson.verification)
    if status == "verified":
        verification["status"] = "verified_target"
        verification.setdefault(
            "explanation",
            "Known verified code target framed for small-model specialist training.",
        )
    return replace(
        lesson,
        verification=verification,
        metadata=metadata,
        training_mode=lesson.training_mode or "verified_fixed_code",
    )


def frame_verified_target_lessons(
    lessons: Iterable[KnowledgeLesson],
    *,
    teacher_source: str = "known_verified_target",
    output_format: str = FIXED_CODE_XML_FORMAT,
    require_verified: bool = True,
) -> list[KnowledgeLesson]:
    """Frame many lessons as verified-output small-model quality targets."""

    return [
        frame_verified_target_lesson(
            lesson,
            teacher_source=teacher_source,
            output_format=output_format,
            require_verified=require_verified,
        )
        for lesson in lessons
    ]
