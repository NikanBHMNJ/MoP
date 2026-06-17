"""Read demo lessons and print verified coding/debugging lessons."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.kts import LessonStore


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GENERATED_PATH = DATA_DIR / "coding_bugfix_lessons.jsonl"
DEMO_PATH = DATA_DIR / "demo_lessons.jsonl"


def main() -> None:
    """Load demo lessons and print filtered JSON records."""

    path = GENERATED_PATH if GENERATED_PATH.exists() else DEMO_PATH
    if not path.exists():
        raise SystemExit(
            "Run python examples/generate_coding_bugfix_lessons.py "
            "or python examples/create_lessons.py first."
        )

    store = LessonStore(path)
    lessons = store.filter(
        domain="coding",
        skill="debugging",
        target_modules=["coding", "debugging"],
        module_match="all",
        verification_status="verified",
    )

    for lesson in lessons:
        print(json.dumps(lesson.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
