"""Build repair lessons from generated-code failures."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from mopforge.formatting import FIXED_CODE_XML_FORMAT
from mopforge.kts import KnowledgeLesson
from mopforge.repair.schema import RepairFailureRecord


def build_repair_lesson_from_failure(
    failure: RepairFailureRecord,
    *,
    repair_id_prefix: str = "repair",
    difficulty_delta: int = 1,
) -> KnowledgeLesson:
    """Create a validated repair lesson from a failed generated candidate.

    Repair lessons use ``skill="repair"`` to distinguish them from ordinary
    debugging lessons. Their target modules include ``coding`` and
    ``debugging``; the separate ``repair`` module is intentionally not added
    until the project defines it as a known module.
    """

    failure.validate()
    source_difficulty = _coerce_difficulty(
        failure.metadata.get("source_difficulty", 3)
    )
    difficulty = max(1, min(5, source_difficulty + difficulty_delta))
    target_modules = _repair_target_modules(failure.target_modules)
    failure_hash = hashlib.sha1(
        (failure.lesson_id + failure.failure_type + failure.candidate_code).encode(
            "utf-8", errors="replace"
        )
    ).hexdigest()[:10]

    return KnowledgeLesson(
        id=f"{_slug(repair_id_prefix)}-{_slug(failure.lesson_id)}-{failure_hash}",
        domain="coding",
        skill="repair",
        subskill=failure.failure_type,
        difficulty=difficulty,
        target_modules=target_modules,
        input=_build_repair_input(failure),
        expected_output=failure.expected_output,
        verification={
            "type": "python_tests",
            "status": "verified_target",
            "explanation": (
                "Expected output is the known verified target from the source "
                "lesson; the generated candidate failed verification."
            ),
            "candidate_failure_type": failure.failure_type,
            "candidate_exit_code": failure.exit_code,
            "candidate_timeout": failure.timeout,
        },
        metadata={
            "source_lesson_id": failure.lesson_id,
            "generated_failure_type": failure.failure_type,
            "candidate_code": failure.candidate_code,
            "generated_text": failure.generated_text,
            "quality_output_format": FIXED_CODE_XML_FORMAT,
            "verified_teacher_target": True,
            "teacher_source": "known_verified_target",
            "repair_generated_from_failure": True,
            "verifier_stdout": failure.verifier_stdout,
            "verifier_stderr": failure.verifier_stderr,
            **_metadata_without_reserved(failure.metadata),
        },
        concept="Repair failed generated code into the known verified solution.",
        common_failures=[failure.failure_type],
        training_mode="repair_from_failure",
        source="mopforge.repair",
    )


def _build_repair_input(failure: RepairFailureRecord) -> str:
    sections = [
        "Repair the generated Python code so it satisfies the original task and tests.",
        "",
        "<original_task>",
        failure.original_input.rstrip(),
        "",
        "<failed_candidate>",
        failure.candidate_code.rstrip() or "<empty candidate>",
        "",
        "<failure_type>",
        failure.failure_type,
    ]
    if failure.verifier_stdout.strip():
        sections.extend(["", "<verifier_stdout>", failure.verifier_stdout.rstrip()])
    if failure.verifier_stderr.strip():
        sections.extend(["", "<verifier_stderr>", failure.verifier_stderr.rstrip()])
    sections.extend(["", "<instruction>", "Return only the repaired Python solution."])
    return "\n".join(sections)


def _repair_target_modules(target_modules: list[str]) -> list[str]:
    ordered = []
    for module in [*target_modules, "coding", "debugging"]:
        if module not in ordered:
            ordered.append(module)
    return ordered


def _coerce_difficulty(value: Any) -> int:
    try:
        difficulty = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, difficulty))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "repair"


def _metadata_without_reserved(metadata: dict[str, Any]) -> dict[str, Any]:
    reserved = {
        "source_lesson_id",
        "generated_failure_type",
        "candidate_code",
        "generated_text",
        "repair_generated_from_failure",
        "verifier_stdout",
        "verifier_stderr",
    }
    return {key: value for key, value in metadata.items() if key not in reserved}
