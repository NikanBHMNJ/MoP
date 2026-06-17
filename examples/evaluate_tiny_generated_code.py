"""CPU smoke evaluation for generated code against lesson tests."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.eval import (
    evaluate_generated_code_for_lesson,
    summarize_generation_results,
    write_generation_eval_results,
)
from mopforge.experiments import TinyExperimentConfig, train_tiny_dense, train_tiny_mop_oracle
from mopforge.kts import LessonStore
from mopforge.tokenization import ByteTokenizer


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "tiny_generated_code_eval.json"
)


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if DATA_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(DATA_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Run tiny generated-code evaluation and write JSON results."""

    print(
        "CPU smoke generated-code evaluation only. "
        "Pass rates are not meaningful model-quality claims."
    )
    ensure_lessons()

    lessons = LessonStore(DATA_PATH).load_all()
    config = TinyExperimentConfig(
        train_steps=3,
        eval_batches=1,
        generation_eval_examples=3,
        max_new_tokens=32,
    )
    tokenizer = ByteTokenizer()
    eval_lessons = lessons[: config.generation_eval_examples]

    dense_model, _ = train_tiny_dense(lessons, config, tokenizer)
    dense_results = [
        evaluate_generated_code_for_lesson(
            dense_model,
            tokenizer,
            lesson,
            max_new_tokens=config.max_new_tokens,
        )
        for lesson in eval_lessons
    ]

    mop_model, _ = train_tiny_mop_oracle(lessons, config, tokenizer)
    mop_results = [
        evaluate_generated_code_for_lesson(
            mop_model,
            tokenizer,
            lesson,
            max_new_tokens=config.max_new_tokens,
            active_modules=list(lesson.target_modules),
        )
        for lesson in eval_lessons
    ]

    report = [
        {
            "model": "tiny_dense",
            "routing": "none",
            "summary": summarize_generation_results(dense_results),
            "results": dense_results,
        },
        {
            "model": "tiny_mop",
            "routing": "oracle",
            "summary": summarize_generation_results(mop_results),
            "results": mop_results,
        },
    ]
    output_path = write_generation_eval_results(report, OUTPUT_PATH)

    for group in report:
        print(f"{group['model']} / {group['routing']}")
        for result in group["results"]:
            print(
                f"  {result['lesson_id']}: "
                f"passed={result['passed']} "
                f"failure_type={result['failure_type']}"
            )
    print(f"Wrote generated-code evaluation JSON to {output_path}")


if __name__ == "__main__":
    main()
