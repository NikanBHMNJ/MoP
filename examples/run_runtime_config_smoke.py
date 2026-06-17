"""Dry-run and execute runtime-aware config templates."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.benchmarks import run_benchmark
from mopforge.configs import benchmark_config_from_envelope, dry_run_config, get_default_config
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.builders import generate_coding_bugfix_lessons


def ensure_tiny_index() -> None:
    lesson_path = Path("data/coding_bugfix_lessons.jsonl")
    indexed_path = Path("data/indexed_lessons.jsonl")
    index_path = Path("data/kts_index.sqlite")
    if indexed_path.exists() and index_path.exists():
        return
    lesson_path.parent.mkdir(parents=True, exist_ok=True)
    if not lesson_path.exists():
        LessonStore(lesson_path).add_many(lesson for lesson in generate_coding_bugfix_lessons(count_per_category=1) if lesson.is_verified)
    indexed = IndexedLessonStore(indexed_path, index_path, allow_duplicate_ids=True)
    for lesson in LessonStore(lesson_path).load_all():
        indexed.add(lesson)


def main() -> None:
    print("Runtime config smoke only. CUDA is optional.")
    ensure_tiny_index()
    config = get_default_config("benchmark_runtime_auto")
    config_path = config.save("outputs/runtime_config_smoke/benchmark_runtime_auto.json")
    dry = dry_run_config(config)
    result = run_benchmark(benchmark_config_from_envelope(config))
    print(f"config_path={config_path}")
    print(f"dry_run_runtime={json.dumps(dry['runtime'], sort_keys=True)}")
    print(f"benchmark_id={result.benchmark_id}")
    print(f"metrics_path={result.metrics_path}")
    print(f"runtime_selected_device={result.metrics['runtime']['selected_device']}")
    print(f"runtime_selected_precision={result.metrics['runtime']['selected_precision']}")


if __name__ == "__main__":
    main()
