"""Run the tiny GPUTrainer smoke path with CPU fallback."""

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
            name="example_gpu_tiny_smoke",
            model_type="mop_oracle",
            max_steps=1,
            micro_batch_size=1,
            eval_every_steps=1,
            eval_batches=1,
            save_every_steps=1,
            log_every_steps=1,
            d_model=16,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
            device="auto",
            precision="auto",
            require_device_available=False,
            max_train_examples=4,
            max_eval_examples=2,
        )
    ).train()
    print(f"run_id={result.run_id}")
    print(f"status={result.status}")
    print(f"result_path={result.artifacts.get('gpu_training_result_json')}")
    print(f"runtime_selected_device={result.runtime_metadata.get('selected_device')}")


if __name__ == "__main__":
    main()
