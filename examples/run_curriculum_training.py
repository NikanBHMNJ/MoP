"""Run tiny curriculum-driven training records."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.repair import build_repair_lessons_from_generation_results, write_repair_lessons
from mopforge.runs import RunRegistry, TinyTrainingRunConfig
from mopforge.tokenization import ByteTokenizer
from mopforge.training import run_tiny_training_from_curriculum


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
EVAL_PATH = ROOT / "outputs" / "tiny_generated_code_eval.json"
RUN_LESSON_PATH = ROOT / "data" / "curriculum_training_lessons.jsonl"
RUN_INDEX_PATH = ROOT / "data" / "curriculum_training_index.sqlite"


def main() -> None:
    """Run two tiny curriculum-training jobs and save records under runs/."""

    print(
        "CPU smoke curriculum-training run only. "
        "Losses/pass rates are not meaningful model-quality claims."
    )
    ensure_bugfix_lessons()
    ensure_repair_lessons()
    build_training_store()

    registry = RunRegistry(ROOT / "runs")
    configs = [
        TinyTrainingRunConfig(
            run_name="dense_balanced",
            model_type="dense",
            curriculum_strategy="balanced",
            lesson_path=str(RUN_LESSON_PATH),
            index_path=str(RUN_INDEX_PATH),
        ),
        TinyTrainingRunConfig(
            run_name="mop_oracle_repair_boosted",
            model_type="mop_oracle",
            curriculum_strategy="repair_boosted",
            lesson_path=str(RUN_LESSON_PATH),
            index_path=str(RUN_INDEX_PATH),
        ),
    ]

    for config in configs:
        record = run_tiny_training_from_curriculum(config, registry=registry)
        print(
            f"run_id={record.run_id} "
            f"model_type={record.model_type} "
            f"strategy={record.curriculum_strategy} "
            f"train_loss={record.metrics['train_loss_last']:.4f} "
            f"eval_loss={record.metrics['eval_loss_mean']:.4f} "
            f"finite={record.metrics['finite']} "
            f"run_json={record.artifacts['run_json']}"
        )


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_repair_lessons() -> None:
    if REPAIR_PATH.exists():
        return
    lessons = LessonStore(BUGFIX_PATH).load_all()
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
    repairs = build_repair_lessons_from_generation_results(
        results,
        {lesson.id: lesson for lesson in lessons},
    )
    write_repair_lessons(repairs, REPAIR_PATH)


def build_training_store() -> None:
    lessons = LessonStore(BUGFIX_PATH).load_all()
    if REPAIR_PATH.exists():
        lessons.extend(LessonStore(REPAIR_PATH).load_all())

    if RUN_LESSON_PATH.exists():
        RUN_LESSON_PATH.unlink()
    if RUN_INDEX_PATH.exists():
        RUN_INDEX_PATH.unlink()

    store = IndexedLessonStore(RUN_LESSON_PATH, RUN_INDEX_PATH)
    for lesson in lessons:
        store.add(lesson)


if __name__ == "__main__":
    main()
