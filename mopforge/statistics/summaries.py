"""Small standard-library statistical summaries."""

from __future__ import annotations

import math
import statistics
from typing import Any


def finite_numbers(values) -> list[float]:
    result = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def mean(values) -> float | None:
    numbers = finite_numbers(values)
    return sum(numbers) / len(numbers) if numbers else None


def stddev(values) -> float | None:
    numbers = finite_numbers(values)
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0 if numbers else None


def stderr(values) -> float | None:
    numbers = finite_numbers(values)
    if not numbers:
        return None
    return (stddev(numbers) or 0.0) / math.sqrt(len(numbers))


def median(values) -> float | None:
    numbers = finite_numbers(values)
    return statistics.median(numbers) if numbers else None


def percent_change(value, baseline) -> float | None:
    numbers = finite_numbers([value, baseline])
    if len(numbers) != 2 or numbers[1] == 0:
        return None
    return (numbers[0] - numbers[1]) / abs(numbers[1]) * 100.0


def summarize_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = finite_numbers(row.get(metric) for row in rows)
    return {
        "metric": metric,
        "count": len(values),
        "mean": mean(values),
        "stddev": stddev(values),
        "stderr": stderr(values),
        "median": median(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_by_group(rows: list[dict[str, Any]], group_by: str, metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_by)), []).append(row)
    output = []
    for group in sorted(groups):
        summary = {"group": group, "count": len(groups[group])}
        for metric in metrics:
            metric_summary = summarize_metric(groups[group], metric)
            for key, value in metric_summary.items():
                if key != "metric":
                    summary[f"{metric}_{key}"] = value
        output.append(summary)
    return output


def compare_groups_simple(rows: list[dict[str, Any]], group_by: str, metric: str) -> dict[str, Any]:
    summaries = summarize_by_group(rows, group_by, [metric])
    baseline = summaries[0] if summaries else None
    comparisons = []
    if baseline is not None:
        base_mean = baseline.get(f"{metric}_mean")
        for summary in summaries:
            comparisons.append(
                {
                    "group": summary["group"],
                    "mean": summary.get(f"{metric}_mean"),
                    "delta_vs_first": None if base_mean is None or summary.get(f"{metric}_mean") is None else summary[f"{metric}_mean"] - base_mean,
                    "percent_change_vs_first": percent_change(summary.get(f"{metric}_mean"), base_mean),
                }
            )
    return {"group_by": group_by, "metric": metric, "summaries": summaries, "comparisons": comparisons}
