"""Run the tiny dense-vs-MoP CPU smoke comparison."""

from __future__ import annotations

from mopforge.experiments import (
    TinyExperimentConfig,
    load_or_generate_lessons,
    run_tiny_comparison,
    write_results,
)


def main() -> None:
    """Run and write the tiny comparison report."""

    config = TinyExperimentConfig()
    print("CPU smoke comparison only. Losses are not meaningful performance claims.")
    lessons = load_or_generate_lessons(config.lesson_path)
    results = run_tiny_comparison(lessons, config)
    json_path, csv_path = write_results(results, "outputs", write_csv=True)

    print("\nmodel       routing          train_loss  eval_loss  finite")
    print("----------  ---------------  ----------  ---------  ------")
    for result in results:
        print(
            f"{result['model']:<10}  "
            f"{result['routing']:<15}  "
            f"{result['train_loss_last']:<10.4f}  "
            f"{result['eval_loss_mean']:<9.4f}  "
            f"{str(result['finite']):<6}"
        )

    print(f"\nWrote JSON results to {json_path}")
    if csv_path is not None:
        print(f"Wrote CSV results to {csv_path}")


if __name__ == "__main__":
    main()
