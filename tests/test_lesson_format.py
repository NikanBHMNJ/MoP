"""Tests for causal-LM lesson formatting."""

from __future__ import annotations

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.formatting import FIXED_CODE_XML_FORMAT, format_lesson_for_causal_lm
from mopforge.quality import frame_verified_target_lesson


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
        "output_format",
    }
    assert formatted["lesson_id"] == lesson.id
    assert formatted["output_format"] == "raw"
    assert formatted["target"] == lesson.expected_output.rstrip() + "\n"
    assert formatted["full_text"] == formatted["prompt"] + formatted["target"]
    assert "<task>" in formatted["prompt"]
    assert lesson.input.rstrip() in formatted["prompt"]
    assert lesson.expected_output.rstrip() not in str(formatted["prompt"])


def test_lesson_formatter_fixed_code_xml_target() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]
    framed = frame_verified_target_lesson(lesson, require_verified=False)

    formatted = format_lesson_for_causal_lm(framed)

    assert formatted["output_format"] == FIXED_CODE_XML_FORMAT
    assert "Return exactly <fixed_code>...</fixed_code>" in str(formatted["prompt"])
    assert formatted["target"] == (
        f"<fixed_code>\n{lesson.expected_output.rstrip()}\n</fixed_code>\n"
    )
    assert formatted["full_text"] == formatted["prompt"] + formatted["target"]
    assert lesson.expected_output.rstrip() not in str(formatted["prompt"])
