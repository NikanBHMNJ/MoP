"""Tests for causal-LM lesson formatting."""

from __future__ import annotations

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.formatting import format_lesson_for_causal_lm


def test_lesson_formatter_output_fields() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]

    formatted = format_lesson_for_causal_lm(lesson)

    assert set(formatted) == {
        "prompt",
        "target",
        "full_text",
        "lesson_id",
        "target_modules",
        "domain",
        "skill",
    }
    assert formatted["lesson_id"] == lesson.id
    assert formatted["target"] == lesson.expected_output.rstrip() + "\n"
    assert formatted["full_text"] == formatted["prompt"] + formatted["target"]
    assert "<task>" in formatted["prompt"]
    assert lesson.input.rstrip() in formatted["prompt"]
    assert lesson.expected_output.rstrip() not in str(formatted["prompt"])
