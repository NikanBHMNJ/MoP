"""Demonstrate config envelopes, validation, dry runs, and a tiny run."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.configs import (
    dry_run_config,
    finetune_config_from_envelope,
    get_default_config,
    validate_config_envelope,
)
from mopforge.kts import IndexedLessonStore, LessonStore
from mopforge.sft import run_finetune


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "cli_config_demo"
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"


def main() -> None:
    """Write default configs, validate/dry-run them, and execute one SFT run."""

    print("CPU smoke CLI config demo only. Metrics are not model-quality claims.")
    ensure_bugfix_lessons()
    ensure_indexed_kts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = {
        "sft_full": get_default_config("sft_full"),
        "sft_generated": get_default_config("sft_generated"),
        "pretrain": get_default_config("pretrain"),
        "trainer": get_default_config("trainer"),
    }
    for name, config in configs.items():
        config.payload["run_registry_root"] = str(OUTPUT_DIR / "runs")
        config.payload["artifact_root"] = str(OUTPUT_DIR / "artifacts")
        path = config.save(OUTPUT_DIR / f"{name}.json")
        validation_messages = validate_config_envelope(config)
        dry_run = dry_run_config(config)
        print(f"config={name} path={path}")
        print(f"  validation_messages={validation_messages or ['valid']}")
        print(f"  runnable_locally={dry_run['runnable_locally']}")

    result = run_finetune(finetune_config_from_envelope(configs["sft_full"]))
    print(f"run_id={result.run_id}")
    print(f"result_path={result.artifacts['finetune_result_json']}")


def ensure_bugfix_lessons() -> None:
    """Create verified coding bugfix lessons if the demo JSONL is missing."""

    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def ensure_indexed_kts() -> None:
    """Rebuild the small indexed KTS used by config examples."""

    lessons = LessonStore(BUGFIX_PATH).load_all()
    if LESSON_PATH.exists():
        LESSON_PATH.unlink()
    lesson_store = LessonStore(LESSON_PATH)
    lesson_store.add_many(lessons)
    IndexedLessonStore(LESSON_PATH, INDEX_PATH, auto_rebuild=True)


if __name__ == "__main__":
    main()
