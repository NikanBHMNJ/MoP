"""Build repair lessons from tiny generated-code evaluation failures."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense
from mopforge.kts import LessonStore
from mopforge.repair import (
    build_repair_lessons_from_generation_results,
    write_repair_lessons,
)
from mopforge.tokenization import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
LESSON_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
EVAL_PATH = ROOT / "outputs" / "tiny_generated_code_eval.json"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if LESSON_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(LESSON_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def load_or_create_generation_results() -> list[dict]:
    """Load existing eval output or create a tiny dense-model failure set."""

    if EVAL_PATH.exists():
        return _flatten_generation_report(json.loads(EVAL_PATH.read_text(encoding="utf-8")))

    lessons = LessonStore(LESSON_PATH).load_all()
    config = TinyExperimentConfig(train_steps=1, max_new_tokens=16)
    tokenizer = ByteTokenizer()
    model, _ = train_tiny_dense(lessons, config, tokenizer)
    results = [
        evaluate_generated_code_for_lesson(
            model,
            tokenizer,
            lesson,
            max_new_tokens=config.max_new_tokens,
        )
        for lesson in lessons[:3]
    ]
    write_generation_eval_results(
        [{"model": "tiny_dense", "routing": "none", "results": results}],
        EVAL_PATH,
    )
    return results


def main() -> None:
    """Build and store repair lessons from generated-code failures."""

    print(
        "CPU smoke repair-loop MVP only. Repair lessons are derived from tiny "
        "model failures and are not a real training dataset yet."
    )
    ensure_lessons()
    lessons = LessonStore(LESSON_PATH).load_all()
    lessons_by_id = {lesson.id: lesson for lesson in lessons}
    generation_results = load_or_create_generation_results()
    failures = [result for result in generation_results if not result.get("passed")]
    repair_lessons = build_repair_lessons_from_generation_results(
        generation_results, lessons_by_id
    )
    written = write_repair_lessons(repair_lessons, REPAIR_PATH)

    print(f"generation_results: {len(generation_results)}")
    print(f"failures: {len(failures)}")
    print(f"repair_lessons_written: {written}")
    if repair_lessons:
        print(f"first_repair_lesson_id: {repair_lessons[0].id}")
        print(f"first_repair_target_modules: {repair_lessons[0].target_modules}")
    print(f"Wrote repair lessons to {REPAIR_PATH}")


def _flatten_generation_report(report: object) -> list[dict]:
    if not isinstance(report, list):
        return []
    if report and isinstance(report[0], dict) and "results" in report[0]:
        flattened: list[dict] = []
        for group in report:
            for result in group.get("results", []):
                enriched = dict(result)
                enriched.setdefault("model", group.get("model"))
                enriched.setdefault("routing", group.get("routing"))
                flattened.append(enriched)
        return flattened
    return [dict(item) for item in report if isinstance(item, dict)]


if __name__ == "__main__":
    main()
