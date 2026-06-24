"""Evaluation helpers for generated code."""

from mopforge.eval.code_extract import extract_python_code, has_complete_fixed_code_block
from mopforge.eval.code_generation import (
    evaluate_candidate_text_for_lesson,
    evaluate_ground_truth_controls,
    evaluate_generated_code,
    evaluate_generated_code_for_lesson,
    select_generation_eval_lessons,
    summarize_generation_results,
    write_generation_eval_results,
)
from mopforge.eval.standard_code import (
    CodeBenchmarkTask,
    audit_code_contamination,
    evaluate_code_completion,
    load_code_benchmark,
    run_code_benchmark,
)

__all__ = [
    "evaluate_candidate_text_for_lesson",
    "evaluate_ground_truth_controls",
    "evaluate_generated_code",
    "evaluate_generated_code_for_lesson",
    "extract_python_code",
    "has_complete_fixed_code_block",
    "select_generation_eval_lessons",
    "summarize_generation_results",
    "write_generation_eval_results",
    "CodeBenchmarkTask",
    "audit_code_contamination",
    "evaluate_code_completion",
    "load_code_benchmark",
    "run_code_benchmark",
]
