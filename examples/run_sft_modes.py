"""Run tiny CPU FT/SFT mode API smoke examples."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.sft import FinetuneConfig, run_finetune


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
RUNS_ROOT = ROOT / "runs"
ARTIFACT_ROOT = ROOT / "artifacts"


def main() -> None:
    """Run several tiny FT/SFT mode examples."""

    print("CPU smoke FT/SFT mode API only. Metrics are not model-quality claims.")
    ensure_bugfix_lessons()
    repair_count = ensure_indexed_kts()

    configs = [
        FinetuneConfig(
            mode="sft_full",
            model_type="dense",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=256,
        ),
        FinetuneConfig(
            mode="sft_module",
            target_modules=["coding"],
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=256,
        ),
        FinetuneConfig(
            mode="sft_adapter",
            target_modules=["coding", "debugging"],
            fast_adapter_names=["coding", "debugging", "repair"],
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=256,
        ),
        FinetuneConfig(
            mode="sft_generated",
            target_modules=["coding", "debugging"],
            generated_condition_names=["coding", "debugging", "repair"],
            generated_rank=4,
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=256,
        ),
    ]
    if repair_count:
        configs.append(
            FinetuneConfig(
                mode="repair_sft",
                model_type="dense",
                lesson_path=str(LESSON_PATH),
                index_path=str(INDEX_PATH),
                run_registry_root=str(RUNS_ROOT),
                artifact_root=str(ARTIFACT_ROOT),
                max_steps=1,
                eval_batches=1,
                batch_size=2,
                max_seq_len=256,
            )
        )
    else:
        print("mode=repair_sft skipped: no repair lessons found")

    for config in configs:
        result = run_finetune(config)
        metrics = result.metrics
        counts = metrics["parameter_counts"]
        policy = metrics["trainable_policy"]["mode"]
        checkpoint_ids = result.trainer_result["final_state"].get(
            "checkpoint_artifacts",
            [],
        )
        print(f"mode={result.mode}")
        print(f"  run_id={result.run_id}")
        print(f"  policy_mode={policy}")
        print(
            "  parameters="
            f"trainable={counts['trainable']} "
            f"frozen={counts['frozen']}"
        )
        print(
            f"  train_loss={metrics['train_loss_last']:.4f} "
            f"eval_loss={metrics['eval_loss_mean']:.4f}"
        )
        print(f"  checkpoint_artifact_ids={checkpoint_ids}")
        print(f"  finetune_result_json={result.artifacts['finetune_result_json']}")


def ensure_bugfix_lessons() -> None:
    """Create verified coding bugfix lessons if needed."""

    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_indexed_kts() -> int:
    """Rebuild indexed lessons from bugfix and optional repair stores."""

    lessons = LessonStore(BUGFIX_PATH).load_all()
    repair_lessons = []
    if REPAIR_PATH.exists():
        repair_lessons = LessonStore(REPAIR_PATH).load_all()
        lessons.extend(repair_lessons)
    if LESSON_PATH.exists():
        LESSON_PATH.unlink()
    lesson_store = LessonStore(LESSON_PATH)
    lesson_store.add_many(lessons)
    IndexedLessonStore(LESSON_PATH, INDEX_PATH, auto_rebuild=True)
    return len(repair_lessons)


if __name__ == "__main__":
    main()
