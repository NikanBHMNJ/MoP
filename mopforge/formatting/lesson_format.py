"""Format KnowledgeLesson records for supervised causal-LM training."""

from __future__ import annotations

from mopforge.kts import KnowledgeLesson


FIXED_CODE_XML_FORMAT = "fixed_code_xml"


def format_lesson_prompt(lesson: KnowledgeLesson) -> str:
    """Return only the deterministic causal-LM prompt for generation."""

    return str(format_lesson_for_causal_lm(lesson)["prompt"])


def format_lesson_for_causal_lm(lesson: KnowledgeLesson) -> dict[str, object]:
    """Return deterministic prompt/target text for one lesson.

    The prompt contains task metadata and context. The target contains only the
    expected output so training code can mask prompt labels cleanly.
    """

    lesson.validate()

    target_modules = ", ".join(lesson.target_modules)
    concept = lesson.concept or "No explicit concept provided."
    subskill = lesson.subskill or "none"
    failures = "\n".join(f"- {failure}" for failure in lesson.common_failures)
    if not failures:
        failures = "- None listed."
    output_format = _output_format(lesson)
    task_text = _task_text(output_format)

    prompt = (
        "<lesson>\n"
        f"domain: {lesson.domain}\n"
        f"skill: {lesson.skill}\n"
        f"subskill: {subskill}\n"
        f"difficulty: {lesson.difficulty}\n"
        f"target_modules: {target_modules}\n\n"
        "<concept>\n"
        f"{concept}\n\n"
        "<common_failures>\n"
        f"{failures}\n\n"
        "<input>\n"
        f"{lesson.input.rstrip()}\n\n"
        "<task>\n"
        f"{task_text}\n\n"
        "<expected_output>\n"
    )
    target = _format_target(lesson.expected_output, output_format)

    return {
        "prompt": prompt,
        "target": target,
        "full_text": prompt + target,
        "lesson_id": lesson.id,
        "target_modules": list(lesson.target_modules),
        "domain": lesson.domain,
        "skill": lesson.skill,
        "output_format": output_format,
    }


def _output_format(lesson: KnowledgeLesson) -> str:
    value = (
        lesson.metadata.get("quality_output_format")
        or lesson.metadata.get("output_format")
        or "raw"
    )
    return str(value).strip() or "raw"


def _task_text(output_format: str) -> str:
    if output_format == FIXED_CODE_XML_FORMAT:
        return (
            "Produce only a verified fixed-code block. Return exactly "
            "<fixed_code>...</fixed_code> with no explanation."
        )
    return "Produce the corrected solution."


def _format_target(expected_output: str, output_format: str) -> str:
    target = expected_output.rstrip()
    if output_format == FIXED_CODE_XML_FORMAT:
        return f"<fixed_code>\n{target}\n</fixed_code>\n"
    return target + "\n"
