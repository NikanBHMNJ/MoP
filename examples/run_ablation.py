"""Run a tiny CPU ablation."""

from __future__ import annotations

from pathlib import Path

from mopforge.ablations import run_ablation
from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.configs import get_default_config
from mopforge.configs.validation import ablation_config_from_envelope
from mopforge.kts import IndexedLessonStore, LessonStore


def ensure_tiny_index() -> None:
    lesson_path = Path("data/coding_bugfix_lessons.jsonl")
    indexed_path = Path("data/indexed_lessons.jsonl")
    index_path = Path("data/kts_index.sqlite")
    if indexed_path.exists() and index_path.exists():
        return
    lesson_path.parent.mkdir(parents=True, exist_ok=True)
    store = LessonStore(lesson_path)
    if not lesson_path.exists():
        for lesson in generate_coding_bugfix_lessons(count_per_category=1):
            store.add(lesson)
    indexed = IndexedLessonStore(indexed_path, index_path, allow_duplicate_ids=True)
    for lesson in LessonStore(lesson_path).load_all():
        indexed.add(lesson)


def main() -> None:
    print("CPU smoke ablation only. Metrics are not model-quality claims.")
    ensure_tiny_index()
    config = ablation_config_from_envelope(get_default_config("ablation_adapter_vs_generated"))
    result = run_ablation(config)
    print(f"ablation_id={result.ablation_id}")
    print(f"status={result.status}")
    print(f"experiment_id={result.experiment_id}")
    print(f"analysis_id={result.analysis_id}")
    print(f"report_path={result.report_path}")


if __name__ == "__main__":
    main()
