"""Run and resume a tiny GPUTrainer checkpoint."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.gpu import GPUTrainer, GPUTrainingConfig
from mopforge.kts import LessonStore


def ensure_lessons() -> None:
    path = Path("data/coding_bugfix_lessons.jsonl")
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    LessonStore(path).add_many(
        lesson for lesson in generate_coding_bugfix_lessons(count_per_category=1) if lesson.is_verified
    )


def main() -> None:
    ensure_lessons()
    first = GPUTrainer(
        GPUTrainingConfig(
            name="example_gpu_resume",
            max_steps=1,
            eval_every_steps=1,
            eval_batches=1,
            save_every_steps=1,
            log_every_steps=1,
            d_model=16,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
            require_device_available=False,
            max_train_examples=4,
        )
    ).train()
    checkpoint = first.artifacts["latest_checkpoint_path"]
    second = GPUTrainer(
        GPUTrainingConfig(
            name="example_gpu_resume_second",
            max_steps=2,
            eval_every_steps=1,
            eval_batches=1,
            save_every_steps=1,
            log_every_steps=1,
            d_model=16,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
            require_device_available=False,
            max_train_examples=4,
            resume_from_checkpoint=checkpoint,
        )
    ).train()
    print(f"first_run_id={first.run_id}")
    print(f"checkpoint={checkpoint}")
    print(f"resumed_run_id={second.run_id}")
    print(f"final_global_steps={second.metrics['global_steps']}")


if __name__ == "__main__":
    main()
