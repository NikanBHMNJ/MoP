"""Generate and store verified Python coding/debugging bug-fix lessons."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import LessonStore


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"
)


def main() -> None:
    """Generate 50 verified lessons and write them to JSONL."""

    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    verified_lessons = [lesson for lesson in lessons if lesson.is_verified]

    if len(verified_lessons) != len(lessons):
        status_counts = Counter(lesson.verification["status"] for lesson in lessons)
        raise SystemExit(
            "Refusing to write unverified demo set. "
            f"Verification status counts: {dict(status_counts)}"
        )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    store = LessonStore(OUTPUT_PATH)
    store.add_many(verified_lessons)

    category_counts = Counter(
        lesson.metadata["bug_type"] for lesson in verified_lessons
    )
    status_counts = Counter(lesson.verification["status"] for lesson in lessons)

    print(f"Wrote {len(verified_lessons)} verified lessons to {OUTPUT_PATH}")
    print("Counts by bug category:")
    for bug_type, count in sorted(category_counts.items()):
        print(f"  {bug_type}: {count}")
    print("Counts by verification status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
