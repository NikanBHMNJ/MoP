"""Load experiment, benchmark, and run result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_experiment_summary(
    experiment_id_or_path: str | Path,
    root: str | Path = "experiments",
) -> list[dict[str, Any]]:
    """Load experiment summary rows by ID, directory, or file path."""

    path = _resolve_experiment_summary_path(experiment_id_or_path, root)
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    data = _read_json(path)
    if isinstance(data, dict):
        rows = data.get("rows")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        return [dict(data)]
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    raise ValueError(f"Experiment summary is not a dictionary/list: {path}")


def load_benchmark_metrics(
    benchmark_id_or_path: str | Path,
    root: str | Path = "benchmarks",
) -> dict[str, Any]:
    """Load benchmark metrics by ID, directory, or metrics file path."""

    path = _resolve_benchmark_metrics_path(benchmark_id_or_path, root)
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark metrics must be a dictionary: {path}")
    return data


def load_run_result(path: str | Path) -> dict[str, Any]:
    """Load one run result JSON file or directory containing a known result file."""

    candidate = Path(path)
    if candidate.is_dir():
        for name in (
            "trainer_result.json",
            "finetune_result.json",
            "continued_pretrain_result.json",
            "metrics.json",
        ):
            child = candidate / name
            if child.exists():
                candidate = child
                break
        else:
            raise FileNotFoundError(f"No supported run result file found in: {path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Run result does not exist: {path}")
    data = _read_json(candidate)
    if not isinstance(data, dict):
        raise ValueError(f"Run result must be a dictionary: {candidate}")
    data.setdefault("_source_path", str(candidate))
    return data


def _resolve_experiment_summary_path(value: str | Path, root: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        if candidate.is_dir():
            for name in ("summary.json", "summary.csv"):
                path = candidate / name
                if path.exists():
                    return path
            raise FileNotFoundError(f"No experiment summary found in: {candidate}")
        return candidate
    directory = Path(root) / str(value)
    for name in ("summary.json", "summary.csv"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Experiment summary not found: {value}")


def _resolve_benchmark_metrics_path(value: str | Path, root: str | Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        if candidate.is_dir():
            path = candidate / "metrics.json"
            if path.exists():
                return path
            raise FileNotFoundError(f"No benchmark metrics.json found in: {candidate}")
        return candidate
    path = Path(root) / str(value) / "metrics.json"
    if path.exists():
        return path
    raise FileNotFoundError(f"Benchmark metrics not found: {value}")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]
