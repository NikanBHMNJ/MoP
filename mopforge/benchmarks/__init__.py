"""Local CPU benchmark/evaluation suite for MoP-Forge."""

from mopforge.benchmarks.config import BENCHMARK_TYPES, BenchmarkConfig
from mopforge.benchmarks.evaluators import (
    evaluate_code_correctness,
    evaluate_composite,
    evaluate_loss,
    evaluate_parameter_efficiency,
    evaluate_router,
)
from mopforge.benchmarks.metrics import (
    count_by_key,
    finite_float,
    flatten_metrics,
    json_safe,
    safe_mean,
    safe_rate,
)
from mopforge.benchmarks.registry import BenchmarkRecord, BenchmarkRegistry
from mopforge.benchmarks.runner import BenchmarkResult, run_benchmark

__all__ = [
    "BENCHMARK_TYPES",
    "BenchmarkConfig",
    "BenchmarkRecord",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "count_by_key",
    "evaluate_code_correctness",
    "evaluate_composite",
    "evaluate_loss",
    "evaluate_parameter_efficiency",
    "evaluate_router",
    "finite_float",
    "flatten_metrics",
    "json_safe",
    "run_benchmark",
    "safe_mean",
    "safe_rate",
]
