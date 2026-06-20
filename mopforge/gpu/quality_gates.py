"""Acceptance gates for GPU efficiency claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mopforge.gpu.compare import extract_run_row, load_run_payloads, resolve_run_dir


DEFAULT_ADAPTER_BASELINE_EVAL_LOSS = 5.165306329727173


def evaluate_efficiency_gates(
    *,
    dense_run: str,
    sparse_run: str,
    gpu_runs_dir: str | Path = "gpu_runs",
    adapter_baseline_eval_loss: float = DEFAULT_ADAPTER_BASELINE_EVAL_LOSS,
    same_quality_eval_delta: float = 0.25,
    generation_pass_delta: float = 0.05,
    vram_target_gb: float | None = None,
) -> dict[str, Any]:
    """Evaluate whether a sparse run can support an efficiency claim."""

    dense_row = extract_run_row(dense_run, gpu_runs_dir=gpu_runs_dir)
    sparse_row = extract_run_row(sparse_run, gpu_runs_dir=gpu_runs_dir)
    dense_dir = resolve_run_dir(dense_run, gpu_runs_dir)
    sparse_dir = resolve_run_dir(sparse_run, gpu_runs_dir)
    dense_metrics, dense_result = load_run_payloads(dense_dir)
    sparse_metrics, sparse_result = load_run_payloads(sparse_dir)
    vram_target = (
        float(vram_target_gb)
        if vram_target_gb is not None
        else _default_vram_target(sparse_metrics)
    )
    gates = [
        _lte_gate(
            "loss_1p5x_improvement",
            sparse_row.get("eval_loss"),
            adapter_baseline_eval_loss / 1.5,
            required=True,
        ),
        _lte_gate(
            "loss_2x_stretch",
            sparse_row.get("eval_loss"),
            adapter_baseline_eval_loss / 2.0,
            required=False,
        ),
        _lte_gate(
            "same_quality_eval_loss",
            sparse_row.get("eval_loss"),
            _add(dense_row.get("eval_loss"), same_quality_eval_delta),
            required=True,
        ),
        _lte_gate(
            "peak_reserved_vram",
            sparse_row.get("peak_reserved_gb"),
            vram_target,
            required=True,
        ),
        _gte_gate(
            "throughput_not_worse_than_dense",
            sparse_row.get("tokens_per_sec"),
            dense_row.get("tokens_per_sec"),
            required=True,
        ),
        _not_in_gate(
            "no_quantization",
            sparse_row.get("selected_precision"),
            {"int8", "int4", "fp8", "quantized"},
            required=True,
        ),
        _generation_gate(
            dense_metrics,
            sparse_metrics,
            max_delta=generation_pass_delta,
        ),
        _sparse_checkpoint_gate(sparse_result, sparse_dir),
        _axis_gate(sparse_row),
    ]
    required = [gate for gate in gates if gate["required"]]
    overall_passed = all(gate["passed"] is True for gate in required)
    return {
        "kind": "gpu_efficiency_gate_report",
        "dense_run": dense_row,
        "sparse_run": sparse_row,
        "thresholds": {
            "adapter_baseline_eval_loss": adapter_baseline_eval_loss,
            "loss_1p5x_eval_threshold": adapter_baseline_eval_loss / 1.5,
            "loss_2x_eval_threshold": adapter_baseline_eval_loss / 2.0,
            "same_quality_eval_delta": same_quality_eval_delta,
            "generation_pass_delta": generation_pass_delta,
            "vram_target_gb": vram_target,
        },
        "gates": gates,
        "overall_passed": overall_passed,
        "unknown_required_gates": [
            gate["name"]
            for gate in required
            if gate["passed"] is None
        ],
        "failed_required_gates": [
            gate["name"]
            for gate in required
            if gate["passed"] is False
        ],
    }


def write_gate_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _lte_gate(name: str, observed, threshold, *, required: bool) -> dict[str, Any]:
    passed = None if observed is None or threshold is None else float(observed) <= float(threshold)
    return {
        "name": name,
        "required": required,
        "direction": "<=",
        "observed": observed,
        "threshold": threshold,
        "passed": passed,
    }


def _gte_gate(name: str, observed, threshold, *, required: bool) -> dict[str, Any]:
    passed = None if observed is None or threshold is None else float(observed) >= float(threshold)
    return {
        "name": name,
        "required": required,
        "direction": ">=",
        "observed": observed,
        "threshold": threshold,
        "passed": passed,
    }


def _not_in_gate(name: str, observed, forbidden: set[str], *, required: bool) -> dict[str, Any]:
    normalized = str(observed or "").lower()
    passed = None if observed is None else normalized not in forbidden
    return {
        "name": name,
        "required": required,
        "direction": "not_in",
        "observed": observed,
        "threshold": sorted(forbidden),
        "passed": passed,
    }


def _generation_gate(
    dense_metrics: dict[str, Any],
    sparse_metrics: dict[str, Any],
    *,
    max_delta: float,
) -> dict[str, Any]:
    dense_rate = _generation_rate(dense_metrics)
    sparse_rate = _generation_rate(sparse_metrics)
    threshold = _add(dense_rate, -max_delta)
    passed = None if sparse_rate is None or threshold is None else sparse_rate >= threshold
    return {
        "name": "generated_code_quality",
        "required": True,
        "direction": ">=",
        "observed": sparse_rate,
        "threshold": threshold,
        "dense_pass_rate": dense_rate,
        "passed": passed,
    }


def _sparse_checkpoint_gate(result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    checkpoint_path = (
        dict(result.get("artifacts") or {}).get("latest_checkpoint_path")
        or dict(result.get("state") or {}).get("latest_checkpoint_path")
    )
    if not checkpoint_path:
        checkpoint_path = _latest_checkpoint(run_dir)
    if not checkpoint_path:
        return {
            "name": "sparse_checkpoint",
            "required": True,
            "direction": "is_trainable_only",
            "observed": None,
            "threshold": True,
            "passed": None,
        }
    path = Path(checkpoint_path)
    if not path.exists():
        return {
            "name": "sparse_checkpoint",
            "required": True,
            "direction": "is_trainable_only",
            "observed": str(path),
            "threshold": True,
            "passed": None,
        }
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        trainable_only = bool(
            payload.get("checkpoint_format") == "mopforge_gpu_train_sparse_v1"
            and payload.get("model_state") is None
            and payload.get("trainable_model_state")
        )
    except Exception:
        trainable_only = None
    return {
        "name": "sparse_checkpoint",
        "required": True,
        "direction": "is_trainable_only",
        "observed": str(path),
        "threshold": True,
        "passed": trainable_only,
    }


def _axis_gate(row: dict[str, Any]) -> dict[str, Any]:
    axes = {
        "vram": row.get("peak_reserved_gb"),
        "trainable_params": row.get("trainable_param_ratio"),
        "checkpoint_delta": row.get("checkpoint_size_mb"),
        "active_flops": row.get("estimated_active_flop_ratio"),
        "wall_clock": row.get("tokens_per_sec"),
    }
    passed = all(value is not None for value in axes.values())
    return {
        "name": "efficiency_axes_reported",
        "required": True,
        "direction": "all_present",
        "observed": axes,
        "threshold": "vram/trainable/checkpoint/active_flops/wall_clock",
        "passed": passed,
    }


def _generation_rate(metrics: dict[str, Any]) -> float | None:
    generation = metrics.get("generation_eval")
    if isinstance(generation, dict):
        if generation.get("gen_verifier_pass_rate") is not None:
            return float(generation["gen_verifier_pass_rate"])
        if generation.get("gen_pass_rate") is not None:
            return float(generation["gen_pass_rate"])
    if metrics.get("gen_verifier_pass_rate") is not None:
        return float(metrics["gen_verifier_pass_rate"])
    if metrics.get("gen_pass_rate") is not None:
        return float(metrics["gen_pass_rate"])
    return None


def _default_vram_target(metrics: dict[str, Any]) -> float:
    mode = str(dict(metrics.get("model") or {}).get("trainable_policy", "")).lower()
    policy_mode = str(dict(metrics.get("config") or {}).get("trainable_policy_mode", "")).lower()
    if "core_frozen" in mode or "core_frozen" in policy_mode:
        return 1.3
    return 1.0


def _latest_checkpoint(run_dir: Path) -> str | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    return str(checkpoints[-1]) if checkpoints else None


def _add(value, delta: float) -> float | None:
    return None if value is None else float(value) + float(delta)
