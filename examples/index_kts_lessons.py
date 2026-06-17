"""Build and query the KTS SQLite metadata index."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, LessonStore


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
REPAIR_PATH = ROOT / "data" / "repair_lessons.jsonl"
COMBINED_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"


def ensure_bugfix_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if BUGFIX_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(BUGFIX_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Build a SQLite metadata index and print small query stats."""

    ensure_bugfix_lessons()
    lessons = LessonStore(BUGFIX_PATH).load_all()
    if REPAIR_PATH.exists():
        lessons.extend(LessonStore(REPAIR_PATH).load_all())

    if COMBINED_PATH.exists():
        COMBINED_PATH.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    store = IndexedLessonStore(COMBINED_PATH, INDEX_PATH)
    for lesson in lessons:
        store.add(lesson)

    print(f"Indexed {store.count()} lessons")
    print(f"skill counts: {store.count_by('skill')}")
    print(f"verification counts: {store.count_by('verification_status')}")
    print(
        "debugging module lessons: "
        f"{store.count(target_modules=['debugging'])}"
    )
    verified_repair_count = store.count(
        skill="repair",
        verification_status="verified_target",
        target_modules=["debugging"],
    )
    verified_repair = store.query(
        skill="repair",
        verification_status="verified_target",
        target_modules=["debugging"],
        limit=5,
    )
    print(f"verified_target repair lessons: {verified_repair_count}")
    if verified_repair:
        print(f"first repair lesson id: {verified_repair[0]['id']}")
    print(f"SQLite index path: {INDEX_PATH}")


if __name__ == "__main__":
    main()
