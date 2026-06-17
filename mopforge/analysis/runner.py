"""Analysis runner for normalized comparison reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopforge.analysis.compare import compare_results
from mopforge.analysis.config import AnalysisConfig
from mopforge.analysis.loading import (
    load_benchmark_metrics,
    load_experiment_summary,
    load_run_result,
)
from mopforge.analysis.normalize import (
    normalize_benchmark_metrics,
    normalize_experiment_rows,
    normalize_run_result,
)
from mopforge.analysis.registry import AnalysisRegistry, _now
from mopforge.analysis.reports import build_markdown_report


@dataclass(slots=True)
class AnalysisResult:
    """Result for one analysis run."""

    analysis_id: str
    status: str
    rows_count: int
    report_path: str
    normalized_results_path: str
    comparison_path: str
    record_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "analysis_id": self.analysis_id,
            "status": self.status,
            "rows_count": self.rows_count,
            "report_path": self.report_path,
            "normalized_results_path": self.normalized_results_path,
            "comparison_path": self.comparison_path,
            "record_path": self.record_path,
        }


def run_analysis(
    config: AnalysisConfig,
    registry_root: str | Path = "reports",
) -> AnalysisResult:
    """Load sources, normalize rows, compare them, and write a report."""

    config = AnalysisConfig.from_dict(config.to_dict())
    root = config.output_root if str(registry_root) == "reports" else registry_root
    registry = AnalysisRegistry(root)
    record = registry.create_analysis(config)
    record.status = "running"
    registry.save_record(record)
    rows: list[dict[str, Any]] = []
    normalized_path = ""
    comparison_path = ""
    report_path = ""
    try:
        rows = _load_rows(config)
        comparison = compare_results(
            rows,
            metrics=config.metrics or None,
            group_by=config.group_by or None,
            rank_by=config.rank_by,
            rank_mode=config.rank_mode,
            baseline_filter=config.baseline_filter or None,
        )
        markdown = build_markdown_report(config, rows, comparison)
        normalized_json = registry.write_normalized_results(record.analysis_id, rows)
        registry.write_normalized_results_csv(record.analysis_id, rows)
        comparison_json = registry.write_comparison(record.analysis_id, comparison)
        registry.write_comparison_csv(record.analysis_id, comparison)
        report = registry.write_report_markdown(record.analysis_id, markdown)
        normalized_path = str(normalized_json)
        comparison_path = str(comparison_json)
        report_path = str(report)
        record.status = "completed"
        record.completed_at = _now()
        record.report_path = report_path
        record.normalized_results_path = normalized_path
        record.comparison_path = comparison_path
        record.metadata["rows_count"] = len(rows)
    except Exception as exc:
        record.status = "failed"
        record.completed_at = _now()
        record.metadata["error"] = str(exc)
    registry.save_record(record)
    return AnalysisResult(
        analysis_id=record.analysis_id,
        status=record.status,
        rows_count=len(rows),
        report_path=report_path or str(registry.analysis_dir(record.analysis_id) / "report.md"),
        normalized_results_path=normalized_path
        or str(registry.analysis_dir(record.analysis_id) / "normalized_results.json"),
        comparison_path=comparison_path or str(registry.analysis_dir(record.analysis_id) / "comparison.json"),
        record_path=str(registry.analysis_dir(record.analysis_id) / "record.json"),
    )


def _load_rows(config: AnalysisConfig) -> list[dict[str, Any]]:
    experiment_root = config.metadata.get("experiment_root", "experiments")
    benchmark_root = config.metadata.get("benchmark_root", "benchmarks")
    rows: list[dict[str, Any]] = []
    for experiment_id in config.experiment_ids:
        summary_rows = load_experiment_summary(experiment_id, root=experiment_root)
        rows.extend(normalize_experiment_rows(summary_rows, source_id=experiment_id))
    for benchmark_id in config.benchmark_ids:
        metrics = load_benchmark_metrics(benchmark_id, root=benchmark_root)
        rows.extend(normalize_benchmark_metrics(metrics, source_id=benchmark_id))
    for run_path in config.run_paths:
        result = load_run_result(run_path)
        rows.append(normalize_run_result(result, source_path=run_path))
    return rows
