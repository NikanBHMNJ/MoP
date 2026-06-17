"""Run tiny trainer smoke tests with module-specific trainable policies."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.training import TinyTrainer, TrainerConfig


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
RUNS_ROOT = ROOT / "runs"
ARTIFACT_ROOT = ROOT / "artifacts"


def main() -> None:
    """Run two target-module policy smoke trainer runs."""

    print(
        "CPU smoke module-specific training policy only. Metrics are not "
        "model-quality claims."
    )
    ensure_bugfix_lessons()
    ensure_indexed_kts()

    for module in ("coding", "debugging"):
        result = TinyTrainer(
            TrainerConfig(
                run_name=f"tiny_policy_{module}",
                model_type="mop_oracle",
                lesson_path=str(LESSON_PATH),
                index_path=str(INDEX_PATH),
                run_registry_root=str(RUNS_ROOT),
                artifact_root=str(ARTIFACT_ROOT),
                trainable_policy_mode="target_modules_only",
                trainable_target_modules=[module],
                max_steps=1,
                eval_interval=1,
                checkpoint_interval=1,
                eval_batches=1,
                batch_size=2,
                max_seq_len=256,
                d_model=32,
                n_layers=1,
                n_heads=2,
            )
        ).train()
        counts = result.metrics["parameter_counts"]
        checkpoint_ids = result.final_state.get("checkpoint_artifacts", [])
        print(f"module={module}")
        print(f"  run_id={result.run_id}")
        print(
            "  parameters="
            f"total={counts['total']} "
            f"trainable={counts['trainable']} "
            f"frozen={counts['frozen']}"
        )
        print(f"  checkpoint_artifact_ids={checkpoint_ids}")
        for summary in result.metrics["parameter_group_summaries"]:
            print(
                "  group="
                f"{summary['name']} "
                f"total={summary['total_params']} "
                f"trainable={summary['trainable_params']} "
                f"frozen={summary['frozen_params']}"
            )


def ensure_bugfix_lessons() -> None:
    """Create verified coding bugfix lessons if the demo JSONL is missing."""

    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_indexed_kts() -> None:
    """Rebuild the small indexed KTS used by trainer examples."""

    lessons = LessonStore(BUGFIX_PATH).load_all()
    if LESSON_PATH.exists():
        LESSON_PATH.unlink()
    lesson_store = LessonStore(LESSON_PATH)
    lesson_store.add_many(lessons)
    IndexedLessonStore(LESSON_PATH, INDEX_PATH, auto_rebuild=True)


if __name__ == "__main__":
    main()
