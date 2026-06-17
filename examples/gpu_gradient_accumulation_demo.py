"""Demonstrate gradient accumulation metadata."""

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
    result = GPUTrainer(
        GPUTrainingConfig(
            name="example_gpu_accumulation",
            max_steps=4,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            eval_every_steps=4,
            eval_batches=1,
            save_every_steps=4,
            log_every_steps=1,
            d_model=16,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
            device="auto",
            precision="auto",
            require_device_available=False,
            max_train_examples=4,
        )
    ).train()
    print(f"run_id={result.run_id}")
    print(f"global_steps={result.metrics['global_steps']}")
    print(f"optimizer_steps={result.metrics['optimizer_steps']}")
    print(f"effective_batch_size={result.metrics['effective_batch_size']}")


if __name__ == "__main__":
    main()
