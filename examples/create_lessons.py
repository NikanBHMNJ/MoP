"""Create a small demo Knowledge Training Store JSONL file."""

from __future__ import annotations

from pathlib import Path

from mopforge.kts import KnowledgeLesson, LessonStore


DEMO_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_lessons.jsonl"


def build_demo_lessons() -> list[KnowledgeLesson]:
    """Return sample coding and debugging lessons for the KTS demo."""

    return [
        KnowledgeLesson(
            id="debug-missing-return-001",
            domain="coding",
            skill="debugging",
            subskill="missing-return",
            difficulty=2,
            target_modules=["coding", "debugging"],
            input=(
                "Fix this Python function so it returns the sum of two numbers:\n"
                "def add(a, b):\n"
                "    a + b"
            ),
            expected_output="def add(a, b):\n    return a + b",
            verification={"type": "python_tests", "status": "verified"},
            metadata={"language": "python", "topic": "functions"},
            concept="Functions must explicitly return computed values.",
            common_failures=["Computing a value without returning it."],
            training_mode="supervised_correction",
            source="demo",
        ),
        KnowledgeLesson(
            id="debug-off-by-one-001",
            domain="coding",
            skill="debugging",
            subskill="off-by-one",
            difficulty=3,
            target_modules=["coding", "debugging", "planning"],
            input=(
                "The loop misses the final index. Correct it:\n"
                "for i in range(len(items) - 1):\n"
                "    process(items[i])"
            ),
            expected_output=(
                "for i in range(len(items)):\n"
                "    process(items[i])"
            ),
            verification={"type": "review", "status": "verified"},
            metadata={"language": "python", "topic": "loops"},
            concept="Loop bounds should cover every intended element.",
            common_failures=["Subtracting one from an exclusive upper bound."],
            training_mode="supervised_correction",
            source="demo",
        ),
        KnowledgeLesson(
            id="debug-keyerror-001",
            domain="coding",
            skill="debugging",
            subskill="dict-defaults",
            difficulty=2,
            target_modules=["coding", "debugging"],
            input=(
                "Avoid KeyError when counting words in a dictionary:\n"
                "counts[word] = counts[word] + 1"
            ),
            expected_output="counts[word] = counts.get(word, 0) + 1",
            verification={"type": "python_tests", "status": "partial"},
            metadata={"language": "python", "topic": "dictionaries"},
            concept="Dictionary lookups need defaults when keys may be absent.",
            common_failures=["Assuming a key already exists before incrementing."],
            training_mode="supervised_correction",
            source="demo",
        ),
        KnowledgeLesson(
            id="debug-exception-message-001",
            domain="coding",
            skill="debugging",
            subskill="exceptions",
            difficulty=1,
            target_modules=["coding", "debugging", "core"],
            input="Improve this vague error: raise ValueError('bad')",
            expected_output="raise ValueError('age must be a non-negative integer')",
            verification={"type": "human_review", "status": "verified"},
            metadata={"language": "python", "topic": "exceptions"},
            concept="Errors should describe the violated expectation.",
            common_failures=["Using vague exception messages."],
            training_mode="style_improvement",
            source="demo",
        ),
    ]


def main() -> None:
    """Write demo lessons to ``data/demo_lessons.jsonl``."""

    if DEMO_PATH.exists():
        DEMO_PATH.unlink()

    store = LessonStore(DEMO_PATH)
    lessons = build_demo_lessons()
    store.add_many(lessons)
    print(f"Wrote {len(lessons)} lessons to {DEMO_PATH}")


if __name__ == "__main__":
    main()
