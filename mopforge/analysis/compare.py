"""Comparison helpers for normalized analysis rows."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_METRICS = [
    "final_eval_loss",
    "eval_loss_mean",
    "final_train_loss",
    "pass_rate",
    "router_exact_match_rate",
    "trainable_ratio",
    "total_params",
    "trainable_params",
    "frozen_params",
]


def compare_results(
    rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
    group_by: list[str] | None = None,
    rank_by: str | None = None,
    rank_mode: str = "min",
    baseline_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare normalized result rows with simple summaries and deltas."""

    if rank_mode not in {"min", "max"}:
        raise ValueError("rank_mode must be 'min' or 'max'.")
    rows = [dict(row) for row in rows]
    metrics_considered = list(metrics or _infer_metrics(rows))
    groups = group_rows(rows, group_by or [])
    grouped_summaries = [
        summarize_group(group_rows_value, metrics_considered, group_key=group_key)
        for group_key, group_rows_value in groups.items()
    ]
    ranked = rank_rows(rows, rank_by, rank_mode) if rank_by else []
    baseline = None
    deltas = []
    if baseline_filter:
        matches = filter_rows(rows, baseline_filter)
        baseline = matches[0] if matches else None
        if baseline is not None:
            for row in rows:
                delta_row = {
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "run_id": row.get("run_id"),
                    "mode": row.get("mode"),
                }
                has_delta = False
                for metric in metrics_considered:
                    delta = numeric_delta(row.get(metric), baseline.get(metric))
                    if delta is not None:
                        delta_row[f"{metric}_delta"] = delta
                        has_delta = True
                if has_delta:
                    deltas.append(delta_row)
    warnings = _metric_warnings(rows, metrics_considered)
    if rank_by and not ranked:
        warnings.append(f"rank_by metric has no numeric values: {rank_by}")
    if baseline_filter and baseline is None:
        warnings.append("baseline_filter did not match any row")
    return {
        "row_count": len(rows),
        "metrics": metrics_considered,
        "group_by": list(group_by or []),
        "grouped_summaries": grouped_summaries,
        "rank_by": rank_by,
        "rank_mode": rank_mode,
        "ranked_rows": ranked,
        "best_row": ranked[0] if ranked else None,
        "baseline_filter": dict(baseline_filter or {}),
        "baseline_row": baseline,
        "deltas_vs_baseline": deltas,
        "warnings": warnings,
    }


def rank_rows(
    rows: list[dict[str, Any]],
    metric: str | None,
    mode: str = "min",
) -> list[dict[str, Any]]:
    """Return rows sorted by a numeric metric."""

    if not metric:
        return []
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")
    ranked = []
    for index, row in enumerate(rows):
        value = _number(row.get(metric))
        if value is not None:
            ranked.append((value, index, dict(row)))
    reverse = mode == "max"
    return [row for _, _, row in sorted(ranked, key=lambda item: (item[0], item[1]), reverse=reverse)]


def filter_rows(rows: list[dict[str, Any]], criteria: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows matching all key/value criteria."""

    return [
        dict(row)
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]


def numeric_delta(value, baseline_value) -> float | None:
    """Return ``value - baseline_value`` for finite numeric inputs."""

    left = _number(value)
    right = _number(baseline_value)
    if left is None or right is None:
        return None
    return left - right


def group_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Group rows by key tuple rendered as a stable label."""

    if not keys:
        return {"all": [dict(row) for row in rows]}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = "|".join(f"{key}={row.get(key)}" for key in keys)
        groups.setdefault(label, []).append(dict(row))
    return dict(sorted(groups.items()))


def summarize_group(
    rows: list[dict[str, Any]],
    metrics: list[str],
    group_key: str = "all",
) -> dict[str, Any]:
    """Summarize a group with counts and numeric metric min/mean/max."""

    summary: dict[str, Any] = {
        "group": group_key,
        "count": len(rows),
        "pass_count": sum(1 for row in rows if row.get("status") == "passed" or row.get("passed") is True),
        "fail_count": sum(1 for row in rows if row.get("status") == "failed" or row.get("passed") is False),
        "error_count": sum(1 for row in rows if row.get("error")),
    }
    for metric in metrics:
        values = [_number(row.get(metric)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        summary[f"{metric}_mean"] = sum(numeric_values) / len(numeric_values)
        summary[f"{metric}_min"] = min(numeric_values)
        summary[f"{metric}_max"] = max(numeric_values)
    return summary


def _infer_metrics(rows: list[dict[str, Any]]) -> list[str]:
    found = []
    for metric in DEFAULT_METRICS:
        if any(_number(row.get(metric)) is not None for row in rows):
            found.append(metric)
    return found


def _metric_warnings(rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    warnings = []
    for metric in metrics:
        if not any(_number(row.get(metric)) is not None for row in rows):
            warnings.append(f"metric has no numeric values: {metric}")
    return warnings


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
