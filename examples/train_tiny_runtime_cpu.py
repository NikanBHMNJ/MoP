"""Run one tiny trainer through the runtime layer."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.training import TinyTrainer, TrainerConfig


def ensure_tiny_index() -> None:
    lesson_path = Path("data/coding_bugfix_lessons.jsonl")
    indexed_path = Path("data/indexed_lessons.jsonl")
    index_path = Path("data/kts_index.sqlite")
    if indexed_path.exists() and index_path.exists():
        return
    lesson_path.parent.mkdir(parents=True, exist_ok=True)
    if not lesson_path.exists():
        store = LessonStore(lesson_path)
        store.add_many(lesson for lesson in generate_coding_bugfix_lessons(count_per_category=1) if lesson.is_verified)
    indexed = IndexedLessonStore(indexed_path, index_path, allow_duplicate_ids=True)
    for lesson in LessonStore(lesson_path).load_all():
        indexed.add(lesson)


def main() -> None:
    print("Runtime trainer smoke only. Metrics are not model-quality claims.")
    ensure_tiny_index()
    result = TinyTrainer(
        TrainerConfig(
            run_name="runtime_auto_trainer_smoke",
            model_type="dense",
            device="auto",
            precision="auto",
            enable_amp=True,
            require_device_available=False,
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=128,
            d_model=16,
            n_layers=1,
            n_heads=2,
        )
    ).train()
    result_path = Path(result.artifacts["trainer_result_json"])
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"run_id={result.run_id}")
    print(f"result_path={result_path}")
    print(f"runtime_selected_device={loaded['metrics']['runtime']['selected_device']}")
    print(f"runtime_selected_precision={loaded['metrics']['runtime']['selected_precision']}")


if __name__ == "__main__":
    main()
