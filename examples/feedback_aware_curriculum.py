"""Build a feedback-aware curriculum plan from tiny generated-code failures."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.curriculum import CurriculumConfig, CurriculumScheduler
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense
from mopforge.feedback import LessonFeedbackStore, feedback_records_from_generation_eval
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.repair import build_repair_lessons_from_generation_results, write_repair_lessons
from mopforge.tokenization import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
EVAL_PATH = ROOT / "outputs" / "tiny_generated_code_eval.json"
COMBINED_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
FEEDBACK_DB_PATH = ROOT / "data" / "lesson_feedback.sqlite"
FEEDBACK_EXPORT_PATH = ROOT / "outputs" / "lesson_feedback_export.json"
PLAN_PATH = ROOT / "outputs" / "curriculum_plan_feedback_weighted.json"


def main() -> None:
    """Create a feedback DB and a feedback-weighted curriculum plan."""

    print(
        "CPU smoke feedback-aware curriculum only. Priorities are based on "
        "tiny model failures and are not quality claims."
    )
    ensure_bugfix_lessons()
    report = ensure_generation_eval()
    ensure_repair_lessons(report)
    store = rebuild_indexed_kts()

    if FEEDBACK_DB_PATH.exists():
        FEEDBACK_DB_PATH.unlink()
    feedback_store = LessonFeedbackStore(FEEDBACK_DB_PATH)
    records = feedback_records_from_generation_eval(report)
    inserted = feedback_store.add_many(records)
    feedback_store.export_json(FEEDBACK_EXPORT_PATH)

    scheduler = CurriculumScheduler(indexed_store=store)
    plan = scheduler.build_plan(
        CurriculumConfig(
            strategy="feedback_weighted",
            batch_size=8,
            feedback_store_path=str(FEEDBACK_DB_PATH),
        )
    )
    plan.save_json(PLAN_PATH)

    print(f"feedback records inserted: {inserted}")
    print(f"failure counts by type: {feedback_store.failure_counts_by_type()}")
    print(f"top_5_prioritized_lesson_ids: {plan.lesson_ids[:5]}")
    print(f"counts_by_skill: {plan.counts_by_skill}")
    print(f"counts_by_verification_status: {plan.counts_by_verification_status}")
    print(f"feedback_db: {FEEDBACK_DB_PATH}")
    print(f"feedback_export_json: {FEEDBACK_EXPORT_PATH}")
    print(f"feedback_weighted_plan_json: {PLAN_PATH}")


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
    lesson_by_id = {lesson.id: lesson for lesson in lessons}
    repairs = build_repair_lessons_from_generation_results(
        _flatten_generation_report(report),
        lesson_by_id,
    )
    write_repair_lessons(repairs, REPAIR_PATH)


def rebuild_indexed_kts() -> IndexedLessonStore:
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
    return store


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
