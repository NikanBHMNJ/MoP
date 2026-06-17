"""Small benchmark metric helpers."""

from __future__ import annotations

import math
from typing import Any


def safe_mean(values) -> float | None:
    """Return the mean of finite numeric values, or ``None``."""

    finite_values = [float(value) for value in values if finite_float(value) is not None]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def safe_rate(numerator, denominator) -> float:
    """Return ``numerator / denominator`` or ``0.0`` for zero denominator."""

    denominator = int(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def count_by_key(items, key: str) -> dict[str, int]:
    """Count dictionaries by one key."""

    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        label = str(value) if value is not None else "none"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metric dictionaries into dotted keys."""

    flat: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                key = f"{prefix}.{child_key}" if prefix else str(child_key)
                visit(key, child_value)
            return
        if isinstance(value, list):
            flat[prefix] = len(value)
            return
        flat[prefix] = json_safe(value)

    visit("", metrics)
    return flat


def finite_float(value) -> float | None:
    """Return a finite float or ``None``."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_safe(value):
    """Return a small JSON-safe representation for scalar metrics."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
