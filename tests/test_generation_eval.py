"""Tests for greedy generation and generated-code evaluation."""

from __future__ import annotations

import json

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.eval import (
    evaluate_candidate_text_for_lesson,
    evaluate_ground_truth_controls,
    evaluate_generated_code_for_lesson,
    extract_python_code,
    select_generation_eval_lessons,
    summarize_generation_results,
    write_generation_eval_results,
)
from mopforge.formatting import format_lesson_prompt
from mopforge.kts import KnowledgeLesson
from mopforge.models import TinyCausalTransformer, TinyMoPCausalTransformer
from mopforge.generation import generate_greedy
from mopforge.tokenization import ByteTokenizer


def test_format_lesson_prompt_excludes_expected_output() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]

    prompt = format_lesson_prompt(lesson)

    assert lesson.input.rstrip() in prompt
    assert lesson.expected_output.rstrip() not in prompt


def test_generate_greedy_dense_returns_string_if_torch_installed() -> None:
    if TinyCausalTransformer is None:
        assert TinyCausalTransformer is None
        return

    tokenizer = ByteTokenizer()
    model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
    )

    generated = generate_greedy(model, tokenizer, "def add", max_new_tokens=3)

    assert isinstance(generated, str)


def test_generate_greedy_mop_returns_string_if_torch_installed() -> None:
    if TinyMoPCausalTransformer is None:
        assert TinyMoPCausalTransformer is None
        return

    tokenizer = ByteTokenizer()
    model = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
    )

    generated = generate_greedy(
        model,
        tokenizer,
        "def add",
        max_new_tokens=3,
        active_modules=["coding", "debugging"],
    )

    assert isinstance(generated, str)


def test_evaluate_generated_code_for_lesson_returns_expected_keys_if_torch_installed() -> None:
    if TinyCausalTransformer is None:
        assert TinyCausalTransformer is None
        return

    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]
    tokenizer = ByteTokenizer()
    model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
    )

    result = evaluate_generated_code_for_lesson(
        model,
        tokenizer,
        lesson,
        max_new_tokens=2,
    )

    assert {
        "lesson_id",
        "passed",
        "failure_type",
        "exit_code",
        "timeout",
        "generated_text",
        "candidate_code",
        "target_modules",
    }.issubset(result)


def test_missing_test_metadata_is_handled_without_crashing() -> None:
    lesson = KnowledgeLesson(
        id="missing-tests",
        domain="coding",
        skill="debugging",
        subskill="returns",
        difficulty=1,
        target_modules=["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "manual", "status": "unverified"},
        metadata={},
    )

    result = evaluate_candidate_text_for_lesson(lesson.expected_output, lesson)

    assert result["passed"] is False
    assert result["failure_type"] == "missing_tests"
    assert result["exit_code"] is None


def test_verifier_integration_passes_when_candidate_matches_expected_output() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]

    result = evaluate_candidate_text_for_lesson(lesson.expected_output, lesson)

    assert result["passed"] is True
    assert result["exact_match"] is True
    assert result["failure_type"] is None
    assert result["exit_code"] == 0


def test_generation_selection_balances_all_bug_categories() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=3, verify=False)

    selected = select_generation_eval_lessons(
        lessons,
        max_lessons=5,
        stratify_by="bug_type",
    )

    assert len(selected) == 5
    assert len({lesson.metadata["bug_type"] for lesson in selected}) == 5


def test_ground_truth_controls_pass_raw_and_fixed_code_xml() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)

    controls = evaluate_ground_truth_controls(lessons)

    assert controls["passed"] is True
    assert controls["examples"] == 5
    assert controls["raw"]["summary"]["gen_verifier_pass_rate"] == 1.0
    assert controls["fixed_code_xml"]["summary"]["gen_fixed_code_complete_rate"] == 1.0


def test_fixed_code_block_extraction_supports_quality_framing() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]
    generated_text = f"notes ignored\n<fixed_code>\n{lesson.expected_output}\n</fixed_code>"

    result = evaluate_candidate_text_for_lesson(generated_text, lesson)

    assert extract_python_code(generated_text) == lesson.expected_output.strip()
    assert result["candidate_code"] == lesson.expected_output.strip()
    assert result["passed"] is True
    assert result["exact_match"] is True
    assert result["fixed_code_block_complete"] is True


def test_generation_eval_writer_outputs_valid_json(tmp_path) -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]
    result = evaluate_candidate_text_for_lesson(lesson.expected_output, lesson)

    output_path = write_generation_eval_results([result], tmp_path / "eval.json")

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded[0]["lesson_id"] == lesson.id
    assert loaded[0]["passed"] is True


def test_generation_summary_separates_exact_and_verifier_pass_rates() -> None:
    summary = summarize_generation_results(
        [
            {"passed": True, "exact_match": True, "failure_type": None},
            {"passed": True, "exact_match": False, "failure_type": None},
            {"passed": False, "exact_match": False, "failure_type": "syntax_error"},
        ]
    )

    assert summary["gen_verifier_pass_rate"] == 2 / 3
    assert summary["gen_exact_match_rate"] == 1 / 3
    assert summary["gen_syntax_pass_rate"] == 2 / 3
    assert summary["gen_compile_pass_rate"] == 2 / 3


def test_generation_summary_reports_complete_xml_and_per_category() -> None:
    summary = summarize_generation_results(
        [
            {
                "passed": True,
                "exact_match": True,
                "fixed_code_block_complete": True,
                "failure_type": None,
                "bug_type": "missing_return",
            },
            {
                "passed": False,
                "exact_match": False,
                "fixed_code_block_complete": False,
                "failure_type": "runtime_error",
                "bug_type": "off_by_one",
            },
        ]
    )

    assert summary["gen_fixed_code_complete_rate"] == 0.5
    assert set(summary["per_category"]) == {"missing_return", "off_by_one"}
    assert summary["per_category"]["missing_return"]["gen_verifier_pass_rate"] == 1.0
