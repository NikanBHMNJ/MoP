"""Run the tiny feedback-weighted retraining loop."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense
from mopforge.feedback import LessonFeedbackStore, feedback_records_from_generation_eval
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.loops import FeedbackRetrainingConfig, run_feedback_retraining_loop
from mopforge.repair import build_repair_lessons_from_generation_results, write_repair_lessons
from mopforge.tokenization import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
EVAL_PATH = ROOT / "outputs" / "tiny_generated_code_eval.json"
COMBINED_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
FEEDBACK_DB_PATH = ROOT / "data" / "lesson_feedback.sqlite"


def main() -> None:
    """Run one CPU-smoke feedback retraining loop."""

    print(
        "CPU smoke feedback-retraining loop only. Changes in pass/fail are "
        "not meaningful model-quality claims."
    )
    ensure_bugfix_lessons()
    report = ensure_generation_eval()
    ensure_repair_lessons(report)
    rebuild_indexed_kts()
    ensure_initial_feedback_db(report)

    result = run_feedback_retraining_loop(
        FeedbackRetrainingConfig(
            lesson_path=str(COMBINED_PATH),
            index_path=str(INDEX_PATH),
            feedback_store_path=str(FEEDBACK_DB_PATH),
            run_registry_root=str(ROOT / "runs"),
            train_steps=3,
            eval_batches=1,
            generation_eval_examples=2,
            max_new_tokens=32,
        )
    )

    print(f"loop_id: {result.loop_id}")
    print(f"training_run_id: {result.train_run_id}")
    print(f"feedback_records_added: {result.feedback_records_added}")
    print(f"pass_count: {result.pass_count}")
    print(f"fail_count: {result.fail_count}")
    print(f"failures_by_type: {result.failures_by_type}")
    for name, path in sorted(result.artifacts.items()):
        print(f"{name}: {path}")


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_generation_eval() -> list[dict]:
    if EVAL_PATH.exists():
        loaded = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, list) else []

    lessons = LessonStore(BUGFIX_PATH).load_all()
    config = TinyExperimentConfig(train_steps=1, eval_batches=1, max_new_tokens=16)
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
    report = [{"model": "tiny_dense", "routing": "none", "results": results}]
    write_generation_eval_results(report, EVAL_PATH)
    return report


def ensure_repair_lessons(report: list[dict]) -> None:
    if REPAIR_PATH.exists():
        return
    lessons = LessonStore(BUGFIX_PATH).load_all()
    repairs = build_repair_lessons_from_generation_results(
        _flatten_generation_report(report),
        {lesson.id: lesson for lesson in lessons},
    )
    write_repair_lessons(repairs, REPAIR_PATH)


def rebuild_indexed_kts() -> None:
    lessons = LessonStore(BUGFIX_PATH).load_all()
    if REPAIR_PATH.exists():
        lessons.extend(LessonStore(REPAIR_PATH).load_all())

    if COMBINED_PATH.exists():
        COMBINED_PATH.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    store = IndexedLessonStore(COMBINED_PATH, INDEX_PATH)
    for lesson in lessons:
        store.add(lesson)


def ensure_initial_feedback_db(report: list[dict]) -> None:
    store = LessonFeedbackStore(FEEDBACK_DB_PATH)
    if store.count() > 0:
        return
    store.add_many(feedback_records_from_generation_eval(report))


def _flatten_generation_report(report: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for group in report:
        results = group.get("results")
        if isinstance(results, list):
            flattened.extend(dict(result) for result in results if isinstance(result, dict))
        elif isinstance(group, dict) and "lesson_id" in group:
            flattened.append(dict(group))
    return flattened


if __name__ == "__main__":
    main()
