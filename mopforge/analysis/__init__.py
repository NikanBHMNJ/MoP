"""Local result analysis and Markdown report generation."""

from mopforge.analysis.compare import (
    compare_results,
    filter_rows,
    group_rows,
    numeric_delta,
    rank_rows,
    summarize_group,
)
from mopforge.analysis.config import AnalysisConfig
from mopforge.analysis.loading import (
    load_benchmark_metrics,
    load_experiment_summary,
    load_run_result,
)
from mopforge.analysis.normalize import (
    NORMALIZED_KEYS,
    normalize_benchmark_metrics,
    normalize_experiment_rows,
    normalize_run_result,
)
from mopforge.analysis.registry import AnalysisRecord, AnalysisRegistry
from mopforge.analysis.reports import build_markdown_report, markdown_table
from mopforge.analysis.runner import AnalysisResult, run_analysis

__all__ = [
    "AnalysisConfig",
    "AnalysisRecord",
    "AnalysisRegistry",
    "AnalysisResult",
    "NORMALIZED_KEYS",
    "build_markdown_report",
    "compare_results",
    "filter_rows",
    "group_rows",
    "load_benchmark_metrics",
    "load_experiment_summary",
    "load_run_result",
    "markdown_table",
    "normalize_benchmark_metrics",
    "normalize_experiment_rows",
    "normalize_run_result",
    "numeric_delta",
    "rank_rows",
    "run_analysis",
    "summarize_group",
]
