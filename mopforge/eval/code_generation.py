"""Generated-code evaluation against lesson tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mopforge.eval.code_extract import extract_python_code, has_complete_fixed_code_block
from mopforge.formatting import format_lesson_prompt
from mopforge.generation import generate_greedy
from mopforge.kts import KnowledgeLesson
from mopforge.verify import verify_python_solution


def evaluate_candidate_text_for_lesson(
    generated_text: str,
    lesson: KnowledgeLesson,
) -> dict[str, Any]:
    """Extract and verify generated text for a single lesson."""

    lesson.validate()
    candidate_code = extract_python_code(generated_text)
    test_code = lesson.metadata.get("test_code")
    base = {
        "lesson_id": lesson.id,
        "generated_text": generated_text,
        "candidate_code": candidate_code,
        "fixed_code_block_complete": has_complete_fixed_code_block(generated_text),
        "exact_match": candidate_code.strip() == lesson.expected_output.strip(),
        "target_modules": list(lesson.target_modules),
        "domain": lesson.domain,
        "skill": lesson.skill,
        "bug_type": lesson.metadata.get("bug_type") or lesson.subskill,
    }
    if not isinstance(test_code, str) or not test_code.strip():
        return {
            **base,
            "passed": False,
            "failure_type": "missing_tests",
            "exit_code": None,
            "timeout": False,
            "verifier_stdout": "",
            "verifier_stderr": "",
            "verifier_duration_ms": None,
        }

    result = verify_python_solution(candidate_code, test_code)
    return {
        **base,
        "passed": result.passed,
        "failure_type": None if result.passed else result.error_type,
        "exit_code": result.exit_code,
        "timeout": result.timeout,
        "verifier_stdout": result.stdout,
        "verifier_stderr": result.stderr,
        "verifier_duration_ms": result.duration_ms,
    }


def evaluate_generated_code_for_lesson(
    model: Any,
    tokenizer: Any,
    lesson: KnowledgeLesson,
    max_new_tokens: int = 128,
    device: str | None = None,
    active_modules: list[str] | None = None,
    active_adapters: list[str] | None = None,
    active_conditions: list[str] | None = None,
) -> dict[str, Any]:
    """Generate, extract, and verify code for one lesson."""

    prompt = format_lesson_prompt(lesson)
    generated_text = generate_greedy(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        active_modules=active_modules,
        active_adapters=active_adapters,
        active_conditions=active_conditions,
    )
    return evaluate_candidate_text_for_lesson(generated_text, lesson)


def evaluate_generated_code(
    model: Any,
    tokenizer: Any,
    lessons: Iterable[KnowledgeLesson],
    max_lessons: int = 5,
    max_new_tokens: int = 128,
    device: str | None = None,
    active_modules: list[str] | None = None,
    active_adapters: list[str] | None = None,
    active_conditions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate generated code for a tiny list of lessons."""

    if type(max_lessons) is not int or max_lessons < 0:
        raise ValueError("max_lessons must be a non-negative integer.")
    results: list[dict[str, Any]] = []
    for index, lesson in enumerate(lessons):
        if index >= max_lessons:
            break
        results.append(
            evaluate_generated_code_for_lesson(
                model,
                tokenizer,
                lesson,
                max_new_tokens=max_new_tokens,
                device=device,
                active_modules=active_modules,
                active_adapters=active_adapters,
                active_conditions=active_conditions,
            )
        )
    return results


def summarize_generation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return small pass/failure metrics for generated-code results."""

    summary = _summarize_generation_results(results)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        category = str(result.get("bug_type") or "unknown")
        grouped.setdefault(category, []).append(result)
    summary["per_category"] = {
        category: _summarize_generation_results(items)
        for category, items in sorted(grouped.items())
    }
    return summary


def _summarize_generation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return non-recursive generation summary metrics."""

    failures = Counter(
        result.get("failure_type") or "passed" for result in results
    )
    pass_count = sum(1 for result in results if result.get("passed"))
    exact_match_count = sum(1 for result in results if result.get("exact_match"))
    complete_fixed_code_count = sum(
        1 for result in results if result.get("fixed_code_block_complete")
    )
    total = len(results)
    syntax_failure_count = int(failures.get("syntax_error", 0))
    syntax_pass_count = max(0, total - syntax_failure_count)
    return {
        "gen_eval_examples": total,
        "gen_pass_count": pass_count,
        "gen_pass_rate": pass_count / total if total else 0.0,
        "gen_verifier_pass_count": pass_count,
        "gen_verifier_pass_rate": pass_count / total if total else 0.0,
        "gen_exact_match_count": exact_match_count,
        "gen_exact_match_rate": exact_match_count / total if total else 0.0,
        "gen_fixed_code_complete_count": complete_fixed_code_count,
        "gen_fixed_code_complete_rate": (
            complete_fixed_code_count / total if total else 0.0
        ),
        "gen_syntax_pass_count": syntax_pass_count,
        "gen_syntax_pass_rate": syntax_pass_count / total if total else 0.0,
        "gen_compile_pass_count": syntax_pass_count,
        "gen_compile_pass_rate": syntax_pass_count / total if total else 0.0,
        "gen_failures_by_type": dict(sorted(failures.items())),
    }


def select_generation_eval_lessons(
    lessons: Iterable[KnowledgeLesson],
    *,
    max_lessons: int,
    stratify_by: str | None = None,
) -> list[KnowledgeLesson]:
    """Select a deterministic, optionally balanced generation-eval subset."""

    values = list(lessons)
    if type(max_lessons) is not int or max_lessons < 0:
        raise ValueError("max_lessons must be a non-negative integer.")
    if stratify_by not in {None, "bug_type", "domain", "skill"}:
        raise ValueError("stratify_by must be bug_type, domain, skill, or None.")
    if max_lessons >= len(values):
        return values
    if stratify_by is None:
        return values[:max_lessons]

    groups: dict[str, list[KnowledgeLesson]] = {}
    for lesson in values:
        if stratify_by == "bug_type":
            key = str(lesson.metadata.get("bug_type") or lesson.subskill or "unknown")
        else:
            key = str(getattr(lesson, stratify_by) or "unknown")
        groups.setdefault(key, []).append(lesson)
    selected: list[KnowledgeLesson] = []
    positions = {key: 0 for key in groups}
    while len(selected) < max_lessons:
        added = False
        for key in sorted(groups):
            position = positions[key]
            if position >= len(groups[key]):
                continue
            selected.append(groups[key][position])
            positions[key] += 1
            added = True
            if len(selected) >= max_lessons:
                break
        if not added:
            break
    return selected


def evaluate_ground_truth_controls(
    lessons: Iterable[KnowledgeLesson],
) -> dict[str, Any]:
    """Verify raw and fixed-code framed ground-truth targets."""

    values = list(lessons)
    raw_results = [
        evaluate_candidate_text_for_lesson(lesson.expected_output, lesson)
        for lesson in values
    ]
    fixed_results = [
        evaluate_candidate_text_for_lesson(
            f"<fixed_code>\n{lesson.expected_output.rstrip()}\n</fixed_code>",
            lesson,
        )
        for lesson in values
    ]
    raw_summary = summarize_generation_results(raw_results)
    fixed_summary = summarize_generation_results(fixed_results)
    return {
        "examples": len(values),
        "passed": bool(
            raw_summary["gen_verifier_pass_count"] == len(values)
            and fixed_summary["gen_verifier_pass_count"] == len(values)
        ),
        "raw": {"summary": raw_summary, "results": raw_results},
        "fixed_code_xml": {"summary": fixed_summary, "results": fixed_results},
    }


def write_generation_eval_results(
    results: list[dict[str, Any]],
    path: str | Path = "outputs/tiny_generated_code_eval.json",
) -> Path:
    """Write generated-code evaluation results to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path
