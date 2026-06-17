"""Tests for repair-loop failure-to-lesson conversion."""

from __future__ import annotations

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import KnowledgeLesson, LessonStore
from mopforge.repair import (
    RepairFailureRecord,
    build_repair_lesson_from_failure,
    build_repair_lessons_from_generation_results,
    failure_record_from_generation_result,
    write_repair_lessons,
)


def _lesson() -> KnowledgeLesson:
    return generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]


def _failed_result(lesson: KnowledgeLesson) -> dict:
    return {
        "lesson_id": lesson.id,
        "passed": False,
        "failure_type": "syntax_error",
        "exit_code": 1,
        "timeout": False,
        "generated_text": "not valid python",
        "candidate_code": "not valid python",
        "target_modules": list(lesson.target_modules),
    }


def test_repair_failure_record_round_trip() -> None:
    record = RepairFailureRecord(
        lesson_id="lesson-1",
        original_input="fix this",
        expected_output="fixed",
        generated_text="bad",
        candidate_code="bad",
        failure_type="syntax_error",
        verifier_stderr="SyntaxError",
        exit_code=1,
        target_modules=["coding", "debugging"],
        metadata={"source_difficulty": 2},
    )

    loaded = RepairFailureRecord.from_dict(record.to_dict())

    assert loaded == record


def test_passed_generation_result_produces_no_failure_record() -> None:
    lesson = _lesson()
    result = {"lesson_id": lesson.id, "passed": True}

    assert failure_record_from_generation_result(result, lesson) is None


def test_failed_generation_result_produces_failure_record() -> None:
    lesson = _lesson()

    record = failure_record_from_generation_result(_failed_result(lesson), lesson)

    assert record is not None
    assert record.lesson_id == lesson.id
    assert record.failure_type == "syntax_error"
    assert record.expected_output == lesson.expected_output


def test_repair_lesson_contains_failure_context_and_validates() -> None:
    lesson = _lesson()
    record = failure_record_from_generation_result(_failed_result(lesson), lesson)
    assert record is not None

    repair_lesson = build_repair_lesson_from_failure(record)

    repair_lesson.validate()
    assert repair_lesson.domain == "coding"
    assert repair_lesson.skill == "repair"
    assert repair_lesson.expected_output == lesson.expected_output
    assert lesson.input.rstrip() in repair_lesson.input
    assert "not valid python" in repair_lesson.input
    assert "syntax_error" in repair_lesson.input
    assert "coding" in repair_lesson.target_modules
    assert "debugging" in repair_lesson.target_modules
    assert repair_lesson.verification["status"] == "verified_target"
    assert repair_lesson.metadata["repair_generated_from_failure"] is True


def test_build_repair_lessons_from_generation_results() -> None:
    lesson = _lesson()

    repair_lessons = build_repair_lessons_from_generation_results(
        [_failed_result(lesson), {"lesson_id": lesson.id, "passed": True}],
        {lesson.id: lesson},
    )

    assert len(repair_lessons) == 1
    assert repair_lessons[0].metadata["source_lesson_id"] == lesson.id


def test_write_repair_lessons_to_jsonl(tmp_path) -> None:
    lesson = _lesson()
    record = failure_record_from_generation_result(_failed_result(lesson), lesson)
    assert record is not None
    repair_lesson = build_repair_lesson_from_failure(record)
    path = tmp_path / "repair_lessons.jsonl"

    count = write_repair_lessons([repair_lesson], path)

    assert count == 1
    loaded = LessonStore(path).load_all()
    assert loaded == [repair_lesson]
