"""Compare GPU run efficiency metrics across MoP-Forge runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMPARISON_FIELDS = [
    "run_id",
    "status",
    "train_loss",
    "eval_loss",
    "tokens_per_sec",
    "samples_per_sec",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "final_reserved_gb",
    "total_params",
    "trainable_params",
    "trainable_param_ratio",
    "active_param_estimate",
    "active_param_ratio",
    "checkpoint_size_mb",
    "routing_mode",
    "active_module_density",
    "active_adapter_density",
    "generated_condition_density",
    "selected_device",
    "selected_precision",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = compare_runs(args.runs, gpu_runs_dir=args.gpu_runs_dir)
    if args.output_json:
        write_json(rows, args.output_json)
    if args.output_csv:
        write_csv(rows, args.output_csv)
    print(format_table(rows))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MoP-Forge GPU run efficiency metrics.")
    parser.add_argument("--runs", nargs="+", required=True, help="GPU run IDs or run directories")
    parser.add_argument("--gpu-runs-dir", default="gpu_runs", help="root containing gpu run directories")
    parser.add_argument("--output-json", help="optional JSON output path")
    parser.add_argument("--output-csv", help="optional CSV output path")
    return parser.parse_args(argv)


def compare_runs(runs: list[str], *, gpu_runs_dir: str | Path = "gpu_runs") -> list[dict[str, Any]]:
    return [extract_run_row(run, gpu_runs_dir=gpu_runs_dir) for run in runs]


def extract_run_row(run: str, *, gpu_runs_dir: str | Path = "gpu_runs") -> dict[str, Any]:
    run_dir = resolve_run_dir(run, gpu_runs_dir)
    metrics, result = load_run_payloads(run_dir)
    runtime = dict(metrics.get("runtime") or result.get("runtime_metadata") or {})
    model = dict(metrics.get("model") or {})
    efficiency = dict(metrics.get("efficiency") or {})
    state = dict(result.get("state") or {})
    artifacts = dict(result.get("artifacts") or {})
    checkpoint_path = (
        artifacts.get("latest_checkpoint_path")
        or state.get("latest_checkpoint_path")
        or _latest_checkpoint(run_dir)
    )
    row = {
        "run_id": result.get("run_id") or run_dir.name,
        "status": metrics.get("status") or result.get("status"),
        "train_loss": metrics.get("latest_train_loss"),
        "eval_loss": metrics.get("latest_eval_loss"),
        "tokens_per_sec": _first(efficiency, metrics, "tokens_per_sec"),
        "samples_per_sec": _first(efficiency, metrics, "samples_per_sec"),
        "peak_allocated_gb": _first(efficiency, metrics, "peak_allocated_gb"),
        "peak_reserved_gb": _first(efficiency, metrics, "peak_reserved_gb"),
        "final_reserved_gb": _first(efficiency, metrics, "final_reserved_gb"),
        "total_params": _first(efficiency, model.get("parameter_counts", {}), "total_params", "total"),
        "trainable_params": _first(
            efficiency, model.get("parameter_counts", {}), "trainable_params", "trainable"
        ),
        "trainable_param_ratio": _first(efficiency, model, "trainable_param_ratio"),
        "active_param_estimate": _first(efficiency, model, "active_param_estimate"),
        "active_param_ratio": _first(efficiency, model, "active_param_ratio"),
        "checkpoint_size_mb": _first(efficiency, {}, "checkpoint_size_mb") or _file_size_mb(
            checkpoint_path
        ),
        "routing_mode": model.get("routing_mode"),
        "active_module_density": _first(efficiency, model, "active_module_density"),
        "active_adapter_density": _first(efficiency, model, "active_adapter_density"),
        "generated_condition_density": _first(efficiency, model, "generated_condition_density"),
        "selected_device": runtime.get("selected_device"),
        "selected_precision": runtime.get("selected_precision"),
    }
    return {field: row.get(field) for field in COMPARISON_FIELDS}


def resolve_run_dir(run: str, gpu_runs_dir: str | Path) -> Path:
    candidate = Path(run)
    if candidate.exists() and candidate.is_dir():
        return candidate
    return Path(gpu_runs_dir) / run


def load_run_payloads(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    result_path = run_dir / "gpu_training_result.json"
    metrics = _read_json(metrics_path)
    result = _read_json(result_path)
    if not metrics and result:
        metrics = dict(result.get("metrics") or {})
    if not result and metrics:
        result = {"run_id": run_dir.name, "metrics": metrics}
    if not metrics and not result:
        raise FileNotFoundError(
            f"Could not find metrics.json or gpu_training_result.json for run: {run_dir}"
        )
    return metrics, result


def write_json(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def format_table(rows: list[dict[str, Any]]) -> str:
    display_fields = [
        "run_id",
        "train_loss",
        "eval_loss",
        "tokens_per_sec",
        "peak_reserved_gb",
        "trainable_param_ratio",
        "active_param_ratio",
        "checkpoint_size_mb",
        "selected_device",
    ]
    widths = {
        field: max(len(field), *(len(_cell(row.get(field))) for row in rows))
        for field in display_fields
    }
    lines = [
        " | ".join(field.ljust(widths[field]) for field in display_fields),
        "-+-".join("-" * widths[field] for field in display_fields),
    ]
    for row in rows:
        lines.append(
            " | ".join(_cell(row.get(field)).ljust(widths[field]) for field in display_fields)
        )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first(
    primary: dict[str, Any], fallback: dict[str, Any], key: str, fallback_key: str | None = None
):
    if key in primary:
        return primary.get(key)
    return fallback.get(fallback_key or key)


def _latest_checkpoint(run_dir: Path) -> str | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    return str(checkpoints[-1]) if checkpoints else None


def _file_size_mb(path: str | None) -> float | None:
    if not path:
        return None
    try:
        candidate = Path(path)
        if not candidate.exists():
            return None
        return round(float(candidate.stat().st_size) / (1024**2), 4)
    except Exception:
        return None


def _cell(value: Any) -> str:
    if value is None:
        return "null"
    return str(value)
