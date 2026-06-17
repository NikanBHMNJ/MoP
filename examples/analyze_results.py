"""Build a tiny local analysis report from experiment and benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.analysis import AnalysisConfig, run_analysis
from mopforge.benchmarks import BenchmarkConfig, BenchmarkRegistry, run_benchmark
from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.configs import default_experiment_adapter_vs_generated_config
from mopforge.experiments import ExperimentRegistry, run_experiment
from mopforge.experiments.matrix import ExperimentConfig
from mopforge.kts import IndexedLessonStore, LessonStore


ROOT = Path(__file__).resolve().parents[1]
BUGFIX_PATH = ROOT / "data" / "coding_bugfix_lessons.jsonl"
LESSON_PATH = ROOT / "data" / "indexed_lessons.jsonl"
INDEX_PATH = ROOT / "data" / "kts_index.sqlite"
EXPERIMENT_ROOT = ROOT / "experiments"
BENCHMARK_ROOT = ROOT / "benchmarks"
REPORT_ROOT = ROOT / "reports"


def main() -> None:
    """Run or reuse tiny sources, then build a Markdown analysis report."""

    print("CPU smoke analysis report only. Metrics are not model-quality claims.")
    ensure_bugfix_lessons()
    ensure_indexed_kts()
    experiment_id = latest_completed_experiment_id() or run_tiny_experiment()
    benchmark_id = latest_completed_benchmark_id() or run_tiny_benchmark()
    config = AnalysisConfig(
        name="example_adapter_benchmark_analysis",
        description="Example local analysis over a tiny experiment and benchmark.",
        experiment_ids=[experiment_id],
        benchmark_ids=[benchmark_id],
        output_root=str(REPORT_ROOT),
        metrics=[
            "final_eval_loss",
            "eval_loss_mean",
            "pass_rate",
            "router_exact_match_rate",
            "trainable_ratio",
            "trainable_params",
        ],
        group_by=["source_type", "mode"],
        rank_by="final_eval_loss",
        rank_mode="min",
        baseline_filter={"mode": "sft_adapter"},
        metadata={
            "experiment_root": str(EXPERIMENT_ROOT),
            "benchmark_root": str(BENCHMARK_ROOT),
        },
    )
    result = run_analysis(config, registry_root=REPORT_ROOT)
    comparison = json.loads(Path(result.comparison_path).read_text(encoding="utf-8"))
    print(f"analysis_id={result.analysis_id}")
    print(f"normalized_results_path={result.normalized_results_path}")
    print(f"comparison_path={result.comparison_path}")
    print(f"report_path={result.report_path}")
    best = comparison.get("best_row")
    if best:
        print(
            "top_ranked="
            f"source={best.get('source_type')} "
            f"mode={best.get('mode')} "
            f"final_eval_loss={best.get('final_eval_loss')}"
        )


def latest_completed_experiment_id() -> str | None:
    records = [
        record
        for record in ExperimentRegistry(EXPERIMENT_ROOT).list_experiments()
        if record.status in {"completed", "completed_with_failures"}
    ]
    return records[-1].experiment_id if records else None


def latest_completed_benchmark_id() -> str | None:
    records = [
        record
        for record in BenchmarkRegistry(BENCHMARK_ROOT).list_benchmarks()
        if record.status == "completed"
    ]
    return records[-1].benchmark_id if records else None


def run_tiny_experiment() -> str:
    envelope = default_experiment_adapter_vs_generated_config()
    payload = dict(envelope.payload)
    for run in payload["runs"]:
        run_payload = dict(run["payload"])
        run_payload.update(
            {
                "lesson_path": str(LESSON_PATH),
                "index_path": str(INDEX_PATH),
                "max_steps": 1,
                "eval_batches": 1,
                "batch_size": 1,
                "max_seq_len": 128,
            }
        )
        run["payload"] = run_payload
    result = run_experiment(
        ExperimentConfig.from_dict(payload),
        registry_root=EXPERIMENT_ROOT,
    )
    return result.experiment_id


def run_tiny_benchmark() -> str:
    result = run_benchmark(
        BenchmarkConfig(
            name="example_analysis_parameter_efficiency",
            benchmark_type="parameter_efficiency",
            model_type="mop_oracle",
            lesson_path=str(LESSON_PATH),
            index_path=str(INDEX_PATH),
            max_seq_len=128,
            target_modules=["coding"],
            use_fast_adapters=True,
            metadata={"trainable_policy_mode": "fast_adapters_only"},
            output_root=str(BENCHMARK_ROOT),
        )
    )
    return result.benchmark_id


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
