"""Register, version, split, and materialize a tiny local dataset."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.datasets import DatasetRegistry, create_dataset_split, write_split_jsonl
from mopforge.kts import LessonStore


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
DATASET_ROOT = ROOT / "datasets"
OUTPUT_ROOT = ROOT / "outputs" / "dataset_demo"


def main() -> None:
    """Run a local dataset registry smoke demo."""

    print("Local dataset registry only. It does not download or host datasets.")
    ensure_bugfix_lessons()
    registry = DatasetRegistry(DATASET_ROOT)
    manifest = registry.register_dataset(
        name="coding_bugfix",
        kind="lessons",
        source_paths=[str(BUGFIX_PATH)],
        dataset_id="coding_bugfix",
        description="Tiny local coding bugfix lessons.",
        tags=["cpu", "smoke", "lessons"],
    )
    snapshot = registry.snapshot_dataset("coding_bugfix")
    split = create_dataset_split(
        snapshot,
        train=0.8,
        eval=0.1,
        test=0.1,
        seed=123,
    )
    train_path = write_split_jsonl(
        snapshot,
        split,
        "train",
        OUTPUT_ROOT / "train.jsonl",
    )
    eval_path = write_split_jsonl(
        snapshot,
        split,
        "eval",
        OUTPUT_ROOT / "eval.jsonl",
    )
    print(f"dataset_id={snapshot.dataset_id}")
    print(f"version_id={snapshot.version_id}")
    print(f"combined_sha256={snapshot.combined_sha256}")
    print(f"stats_records={snapshot.stats.record_count}")
    print(f"stats_domains={snapshot.stats.domains}")
    print(f"split_id={split.split_id}")
    print(f"split_counts={split.counts}")
    print(f"train_path={train_path}")
    print(f"eval_path={eval_path}")
    print(f"first_version_id={manifest.version_id}")


def ensure_bugfix_lessons() -> None:
    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


if __name__ == "__main__":
    main()
