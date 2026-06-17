"""Format KnowledgeLesson records for supervised causal-LM training."""

from __future__ import annotations

from mopforge.kts import KnowledgeLesson


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
        "Produce the corrected solution.\n\n"
        "<expected_output>\n"
    )
    target = lesson.expected_output.rstrip() + "\n"

    return {
        "prompt": prompt,
        "target": target,
        "full_text": prompt + target,
        "lesson_id": lesson.id,
        "target_modules": list(lesson.target_modules),
        "domain": lesson.domain,
        "skill": lesson.skill,
    }
