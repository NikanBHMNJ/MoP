"""Format and tokenize generated coding/debugging lessons."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import LessonCausalLMDataset
from mopforge.formatting import format_lesson_for_causal_lm
from mopforge.kts import LessonStore
from mopforge.tokenization import ByteTokenizer


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if DATA_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(DATA_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Print one formatted and tokenized sample."""

    ensure_lessons()
    lessons = LessonStore(DATA_PATH).load_all()
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=512)

    formatted = format_lesson_for_causal_lm(lessons[0])
    item = dataset[0]
    label_mask_count = sum(1 for label in item["labels"] if label == -100)

    print(f"lesson_id: {item['lesson_id']}")
    print(f"target_modules: {', '.join(item['target_modules'])}")
    print(f"input_length: {len(item['input_ids'])}")
    print(f"label_mask_count: {label_mask_count}")
    print("\nPrompt preview:")
    print(str(formatted["prompt"])[:500])
    print("Target:")
    print(str(formatted["target"]))
    print("First 32 token ids:")
    print(item["input_ids"][:32])


if __name__ == "__main__":
    main()
