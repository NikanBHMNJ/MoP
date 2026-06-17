"""Run the CPU-first TinyTrainer skeleton."""

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
    """Run dense and oracle TinyMoP trainer smoke examples."""

    print(
        "CPU smoke trainer skeleton only. Metrics are not meaningful "
        "model-quality claims."
    )
    ensure_bugfix_lessons()
    ensure_indexed_kts()

    configs = [
        TrainerConfig(
            run_name="tiny_trainer_dense",
            model_type="dense",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=2,
            eval_interval=1,
            checkpoint_interval=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=256,
            d_model=32,
            n_layers=1,
            n_heads=2,
        ),
        TrainerConfig(
            run_name="tiny_trainer_mop_oracle",
            model_type="mop_oracle",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            run_registry_root=str(RUNS_ROOT),
            artifact_root=str(ARTIFACT_ROOT),
            max_steps=2,
            eval_interval=1,
            checkpoint_interval=1,
            eval_batches=1,
            batch_size=2,
            max_seq_len=256,
            d_model=32,
            n_layers=1,
            n_heads=2,
        ),
    ]

    for config in configs:
        result = TinyTrainer(config).train()
        checkpoint_ids = result.final_state.get("checkpoint_artifacts", [])
        checkpoint_paths = result.artifacts.get("checkpoint_paths", [])
        print(
            f"run_id={result.run_id} model_type={result.model_type} "
            f"train_loss={result.metrics['train_loss_last']:.4f} "
            f"eval_loss={result.metrics['eval_loss_mean']:.4f} "
            f"finite={result.finite}"
        )
        print(f"  checkpoint_artifact_ids={checkpoint_ids}")
        print(f"  trainer_result_path={result.artifacts['trainer_result_json']}")
        if checkpoint_paths:
            print(f"  latest_checkpoint_path={checkpoint_paths[-1]}")


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_indexed_kts() -> None:
    lessons = LessonStore(BUGFIX_PATH).load_all()
    if LESSON_PATH.exists():
        LESSON_PATH.unlink()
    store = IndexedLessonStore(LESSON_PATH, INDEX_PATH)
    for lesson in lessons:
        store.add(lesson)


if __name__ == "__main__":
    main()
