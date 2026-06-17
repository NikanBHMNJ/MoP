"""Build and consume a local module-specific training queue."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.curriculum import CurriculumConfig
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense
from mopforge.feedback import LessonFeedbackStore, feedback_records_from_generation_eval
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.queues import (
    TrainingQueueStore,
    build_module_queue_from_indexed_store,
    consume_queue_once,
)
from mopforge.repair import build_repair_lessons_from_generation_results, write_repair_lessons
from mopforge.tokenization import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
EVAL_PATH = ROOT / "outputs" / "tiny_generated_code_eval.json"
COMBINED_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
FEEDBACK_DB_PATH = ROOT / "data" / "lesson_feedback.sqlite"
QUEUE_DB_PATH = ROOT / "data" / "training_queue.sqlite"
QUEUE_EXPORT_PATH = ROOT / "outputs" / "training_queue_export.json"


def main() -> None:
    """Build a module queue and run one local smoke consume."""

    print(
        "CPU smoke module-queue MVP only. Queue items are local scheduling "
        "metadata, not production jobs."
    )
    ensure_bugfix_lessons()
    report = ensure_generation_eval()
    ensure_repair_lessons(report)
    indexed_store = rebuild_indexed_kts()
    feedback_store = ensure_feedback_store(report)

    if QUEUE_DB_PATH.exists():
        QUEUE_DB_PATH.unlink()
    queue_store = TrainingQueueStore(QUEUE_DB_PATH)
    strategy = "feedback_weighted" if feedback_store.count() else "repair_boosted"
    config = CurriculumConfig(
        strategy=strategy,
        batch_size=8,
        feedback_store_path=str(FEEDBACK_DB_PATH) if feedback_store.count() else None,
    )
    items = build_module_queue_from_indexed_store(
        indexed_store,
        config,
        modules=["coding", "debugging", "repair"],
        feedback_store=feedback_store if feedback_store.count() else None,
    )
    queue_store.add_many(items)
    before_status = queue_store.counts_by_status()
    consume_result = consume_queue_once(
        queue_store,
        run_registry_root=str(ROOT / "runs"),
    )
    after_status = queue_store.counts_by_status()
    queue_store.export_json(QUEUE_EXPORT_PATH)

    print(f"total queue items: {queue_store.count()}")
    print(f"counts by module: {queue_store.counts_by_module()}")
    print(f"counts by status before consume: {before_status}")
    print(f"counts by status after consume: {after_status}")
    print(f"consumed item ID: {consume_result['item_id']}")
    print(f"consume result: {consume_result}")
    print(f"queue_db: {QUEUE_DB_PATH}")
    print(f"queue_export_json: {QUEUE_EXPORT_PATH}")


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


def ensure_feedback_store(report: list[dict]) -> LessonFeedbackStore:
    store = LessonFeedbackStore(FEEDBACK_DB_PATH)
    if store.count() == 0:
        store.add_many(feedback_records_from_generation_eval(report))
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
