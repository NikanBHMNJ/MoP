"""Run tiny local benchmark suite examples."""

from __future__ import annotations

from pathlib import Path

from mopforge.benchmarks import BenchmarkConfig, run_benchmark
from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.kts import IndexedLessonStore, LessonStore


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
BENCHMARK_ROOT = ROOT / "benchmarks"


def main() -> None:
    """Run a few CPU-safe benchmark smoke paths."""

    print("CPU smoke benchmark suite only. Metrics are not model-quality claims.")
    ensure_bugfix_lessons()
    ensure_indexed_kts()

    configs = [
        BenchmarkConfig(
            name="example_parameter_efficiency",
            benchmark_type="parameter_efficiency",
            model_type="mop_oracle",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            max_seq_len=128,
            target_modules=["coding"],
            use_fast_adapters=True,
            metadata={"trainable_policy_mode": "fast_adapters_only"},
            output_root=str(BENCHMARK_ROOT),
        ),
        BenchmarkConfig(
            name="example_loss",
            benchmark_type="loss",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            max_examples=4,
            batch_size=2,
            max_seq_len=128,
            output_root=str(BENCHMARK_ROOT),
        ),
        BenchmarkConfig(
            name="example_code_correctness",
            benchmark_type="code_correctness",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            max_examples=2,
            generation_examples=2,
            generation_max_new_tokens=32,
            max_seq_len=128,
            output_root=str(BENCHMARK_ROOT),
        ),
    ]

    for config in configs:
        result = run_benchmark(config)
        metrics = result.metrics
        print(f"benchmark_id={result.benchmark_id} type={result.benchmark_type}")
        print(f"  status={result.status}")
        print(f"  metrics_path={result.metrics_path}")
        print(f"  metrics_csv_path={result.metrics_csv_path}")
        if result.examples_path:
            print(f"  examples_path={result.examples_path}")
        if result.benchmark_type == "parameter_efficiency":
            print(
                "  params="
                f"total={metrics['total_params']} "
                f"trainable={metrics['trainable_params']} "
                f"frozen={metrics['frozen_params']}"
            )
        elif result.benchmark_type == "loss":
            print(f"  eval_loss_mean={metrics['eval_loss_mean']}")
        elif result.benchmark_type == "code_correctness":
            print(
                f"  pass_count={metrics['pass_count']} "
                f"fail_count={metrics['fail_count']} "
                f"pass_rate={metrics['pass_rate']:.3f}"
            )


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
