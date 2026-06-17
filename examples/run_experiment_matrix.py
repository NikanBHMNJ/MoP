"""Run a tiny local experiment matrix/list smoke example."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.configs import (
    MoPForgeConfig,
    default_experiment_adapter_vs_generated_config,
    experiment_config_from_envelope,
)
from mopforge.experiments import run_experiment
from mopforge.kts import IndexedLessonStore, LessonStore


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
CONFIG_PATH = ROOT / "configs" / "examples" / "experiment_adapter_vs_generated.json"
EXPERIMENT_ROOT = ROOT / "experiments"


def main() -> None:
    """Run the tiny adapter-vs-generated experiment."""

    print("CPU smoke experiment matrix only. Metrics are not model-quality claims.")
    ensure_bugfix_lessons()
    ensure_indexed_kts()
    config = load_or_default_config()
    result = run_experiment(config, registry_root=EXPERIMENT_ROOT)

    child_run_ids = [
        str(record.get("run_id"))
        for record in result.run_records
        if record.get("run_id")
    ]
    print(f"experiment_id={result.experiment_id}")
    print(f"status={result.status}")
    print(f"total_runs={result.total_runs}")
    print(f"completed_runs={result.completed_runs}")
    print(f"failed_runs={result.failed_runs}")
    print(f"child_run_ids={child_run_ids}")
    print(f"summary_json={result.summary_path}")
    print(f"summary_csv={result.summary_csv_path}")


def load_or_default_config():
    if CONFIG_PATH.exists():
        return experiment_config_from_envelope(MoPForgeConfig.load(CONFIG_PATH))
    return experiment_config_from_envelope(default_experiment_adapter_vs_generated_config())


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_indexed_kts() -> None:
    if LESSON_PATH.exists() and INDEX_PATH.exists():
        return
    lessons = LessonStore(BUGFIX_PATH).load_all()
    store = IndexedLessonStore(LESSON_PATH, INDEX_PATH)
    for lesson in lessons:
        store.add(lesson)


if __name__ == "__main__":
    main()
