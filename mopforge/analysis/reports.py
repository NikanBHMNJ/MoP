"""Markdown report generation for local analyses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mopforge.analysis.config import AnalysisConfig


def build_markdown_report(
    config: AnalysisConfig,
    rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    """Build a deterministic, readable Markdown analysis report."""

    lines = [
        f"# {config.name}",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
    ]
    if config.description:
        lines.extend([config.description, ""])
    lines.extend(
        [
            "## Sources",
            "",
            f"- Experiments: {', '.join(config.experiment_ids) if config.experiment_ids else 'none'}",
            f"- Benchmarks: {', '.join(config.benchmark_ids) if config.benchmark_ids else 'none'}",
            f"- Run paths: {', '.join(config.run_paths) if config.run_paths else 'none'}",
            "",
            "## Normalized Results",
            "",
            f"Normalized row count: {len(rows)}",
            "",
        ]
    )
    preview_columns = [
        "source_type",
        "source_id",
        "kind",
        "mode",
        "model_type",
        "final_eval_loss",
        "eval_loss_mean",
        "pass_rate",
        "trainable_ratio",
    ]
    lines.append(markdown_table(rows[:12], preview_columns))
    lines.append("")
    if config.rank_by:
        lines.extend(
            [
                "## Ranking",
                "",
                f"Ranked by `{config.rank_by}` ({config.rank_mode}).",
                "",
                markdown_table(
                    comparison.get("ranked_rows", [])[:10],
                    ["source_type", "source_id", "run_id", "mode", config.rank_by],
                ),
                "",
            ]
        )
    lines.extend(["## Group Summaries", ""])
    group_rows = comparison.get("grouped_summaries", [])
    group_columns = _group_columns(group_rows)
    lines.append(markdown_table(group_rows, group_columns))
    lines.append("")
    lines.extend(["## Benchmark Metric Highlights", ""])
    highlights = [
        row
        for row in rows
        if row.get("source_type") == "benchmark"
        and any(
            row.get(key) is not None
            for key in ("eval_loss_mean", "pass_rate", "router_exact_match_rate", "trainable_ratio")
        )
    ]
    lines.append(
        markdown_table(
            highlights[:12],
            [
                "source_id",
                "mode",
                "eval_loss_mean",
                "pass_rate",
                "router_exact_match_rate",
                "trainable_ratio",
                "trainable_params",
            ],
        )
    )
    lines.append("")
    if config.baseline_filter:
        lines.extend(["## Baseline Deltas", ""])
        lines.append(
            markdown_table(
                comparison.get("deltas_vs_baseline", [])[:12],
                _delta_columns(comparison.get("deltas_vs_baseline", [])),
            )
        )
        lines.append("")
    warnings = comparison.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(
        [
            "## Limitations",
            "",
            "- CPU smoke analysis report only. Metrics are not model-quality claims.",
            "- No statistical significance, confidence intervals, plotting, or PDF generation.",
            "- Metrics are only as meaningful as the source experiment and benchmark artifacts.",
            "- Local filesystem artifacts are the only supported source in this MVP.",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render a small GitHub-flavored Markdown table."""

    columns = [column for column in columns if column]
    if not columns:
        columns = ["value"]
    header = "| " + " | ".join(_escape(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, separator])
    body = []
    for row in rows:
        body.append("| " + " | ".join(_escape(_format_cell(row.get(column))) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def _group_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["group", "count", "pass_count", "fail_count", "error_count"]
    keys = {key for row in rows for key in row}
    columns = [key for key in preferred if key in keys]
    columns.extend(sorted(keys - set(columns))[:8])
    return columns or preferred


def _delta_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["source_type", "source_id", "run_id", "mode"]
    keys = {key for row in rows for key in row}
    columns = [key for key in preferred if key in keys]
    columns.extend(sorted(key for key in keys if key.endswith("_delta"))[:8])
    return columns or preferred


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {value[key]}" for key in sorted(value)[:3]) + "}"
    if isinstance(value, list):
        return str(len(value))
    return str(value)


def _escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
