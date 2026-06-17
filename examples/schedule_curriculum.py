"""Build deterministic curriculum plans from the indexed KTS."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.curriculum import CurriculumConfig, CurriculumScheduler
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.repair import build_repair_lessons_from_generation_results, write_repair_lessons
from mopforge.tokenization import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
EVAL_PATH = ROOT / "outputs" / "tiny_generated_code_eval.json"
COMBINED_PATH = ROOT / "data" / "curriculum_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "curriculum_index.sqlite"
BALANCED_PATH = ROOT / "outputs" / "curriculum_plan_balanced.json"
REPAIR_PATH_OUT = ROOT / "outputs" / "curriculum_plan_repair_boosted.json"


def main() -> None:
    """Build and save a few small curriculum plans."""

    ensure_bugfix_lessons()
    ensure_repair_lessons()

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

    scheduler = CurriculumScheduler(indexed_store=store)
    balanced = scheduler.build_plan(CurriculumConfig(strategy="balanced", batch_size=8))
    module_targeted = scheduler.build_plan(
        CurriculumConfig(
            strategy="module_targeted",
            target_modules=["debugging"],
            batch_size=8,
        )
    )
    repair_boosted = scheduler.build_plan(
        CurriculumConfig(strategy="repair_boosted", batch_size=8)
    )

    balanced.save_json(BALANCED_PATH)
    repair_boosted.save_json(REPAIR_PATH_OUT)

    print(f"total indexed lessons: {store.count()}")
    for name, plan in [
        ("balanced", balanced),
        ("module_targeted_debugging", module_targeted),
        ("repair_boosted", repair_boosted),
    ]:
        print(f"\n{name}")
        print(f"total: {plan.total}")
        print(f"first_5: {plan.lesson_ids[:5]}")
        print(f"counts_by_skill: {plan.counts_by_skill}")
        print(
            "counts_by_verification_status: "
            f"{plan.counts_by_verification_status}"
        )
    print(f"\nWrote balanced plan to {BALANCED_PATH}")
    print(f"Wrote repair-boosted plan to {REPAIR_PATH_OUT}")


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_repair_lessons() -> None:
    if REPAIR_PATH.exists():
        return
    lessons = LessonStore(BUGFIX_PATH).load_all()
    results = load_or_create_generation_results(lessons)
    repairs = build_repair_lessons_from_generation_results(
        results,
        {lesson.id: lesson for lesson in lessons},
    )
    write_repair_lessons(repairs, REPAIR_PATH)


def load_or_create_generation_results(lessons) -> list[dict]:
    if EVAL_PATH.exists():
        return flatten_generation_report(json.loads(EVAL_PATH.read_text(encoding="utf-8")))

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


def flatten_generation_report(report: object) -> list[dict]:
    if not isinstance(report, list):
        return []
    if report and isinstance(report[0], dict) and "results" in report[0]:
        flattened = []
        for group in report:
            for result in group.get("results", []):
                flattened.append(dict(result))
        return flattened
    return [dict(item) for item in report if isinstance(item, dict)]


if __name__ == "__main__":
    main()
