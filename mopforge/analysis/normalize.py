"""Normalize result artifacts into stable comparison rows."""

from __future__ import annotations

import math
from typing import Any

from mopforge.benchmarks.metrics import json_safe


NORMALIZED_KEYS = [
    "source_type",
    "source_id",
    "run_id",
    "kind",
    "mode",
    "model_type",
    "routing",
    "trainable_policy_mode",
    "use_fast_adapters",
    "use_generated_params",
    "total_params",
    "trainable_params",
    "frozen_params",
    "trainable_ratio",
    "final_train_loss",
    "final_eval_loss",
    "eval_loss_mean",
    "pass_rate",
    "router_exact_match_rate",
    "runtime_selected_device",
    "runtime_selected_precision",
    "runtime_amp_enabled",
    "runtime_gpu_name",
    "finite",
    "result_path",
    "error",
    "metadata",
]


def normalize_experiment_rows(
    rows: list[dict[str, Any]],
    source_id: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize rows from an experiment summary."""

    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = _empty_row()
        output.update(
            {
                "source_type": "experiment",
                "source_id": source_id or row.get("experiment_id"),
                "run_id": row.get("run_id"),
                "kind": row.get("kind"),
                "mode": row.get("mode"),
                "model_type": row.get("model_type"),
                "routing": row.get("routing") or _routing_from_model(row.get("model_type")),
                "trainable_policy_mode": row.get("trainable_policy_mode"),
                "final_train_loss": _number(row.get("final_train_loss")),
                "final_eval_loss": _number(row.get("final_eval_loss")),
                "eval_loss_mean": _number(row.get("eval_loss_mean") or row.get("final_eval_loss")),
                "finite": _bool_or_none(row.get("finite")),
                "runtime_selected_device": row.get("runtime_selected_device"),
                "runtime_selected_precision": row.get("runtime_selected_precision"),
                "runtime_amp_enabled": _bool_or_none(row.get("runtime_amp_enabled")),
                "runtime_gpu_name": row.get("runtime_gpu_name"),
                "result_path": row.get("result_path"),
                "error": row.get("error"),
                "metadata": _metadata(row, exclude=set(NORMALIZED_KEYS) | {"experiment_id"}),
            }
        )
        _merge_parameter_fields(output, row)
        normalized.append(_json_safe_row(output))
    return normalized


def normalize_benchmark_metrics(
    metrics: dict[str, Any],
    source_id: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize benchmark metrics, including composite nested metrics."""

    rows: list[dict[str, Any]] = []
    if not isinstance(metrics, dict):
        return rows
    benchmark_id = source_id or metrics.get("benchmark_id")
    base = _benchmark_row(metrics, benchmark_id, component=None)
    for component in ("parameter_efficiency", "loss", "code_correctness", "router"):
        nested = metrics.get(component)
        if isinstance(nested, dict):
            _merge_metric_fields(base, nested)
            if component == "router":
                base["router_exact_match_rate"] = _number(
                    nested.get("router_exact_match_rate") or nested.get("exact_match_rate")
                )
    rows.append(_json_safe_row(base))
    for component in ("parameter_efficiency", "loss", "code_correctness", "router"):
        nested = metrics.get(component)
        if isinstance(nested, dict):
            rows.append(_json_safe_row(_benchmark_row(nested, benchmark_id, component=component)))
    return rows


def normalize_run_result(
    result: dict[str, Any],
    source_path: str | None = None,
) -> dict[str, Any]:
    """Normalize a trainer, finetune, pretrain, or metrics result dictionary."""

    metrics = _result_metrics(result)
    config = _result_config(result, metrics)
    output = _empty_row()
    source = source_path or result.get("_source_path")
    mode = result.get("mode") or metrics.get("finetune_mode") or config.get("mode")
    kind = _kind_from_result(result, metrics, config, source)
    policy = metrics.get("trainable_policy") if isinstance(metrics, dict) else None
    output.update(
        {
            "source_type": "run",
            "source_id": source,
            "run_id": result.get("run_id") or metrics.get("run_id"),
            "kind": kind,
            "mode": mode,
            "model_type": result.get("model_type") or metrics.get("model_type") or config.get("model_type"),
            "routing": result.get("routing_mode") or metrics.get("routing_mode"),
            "trainable_policy_mode": (
                policy.get("mode") if isinstance(policy, dict) else metrics.get("trainable_policy_mode")
            ),
            "use_fast_adapters": config.get("use_fast_adapters"),
            "use_generated_params": config.get("use_generated_params"),
            "final_train_loss": _number(
                metrics.get("train_loss_last")
                or metrics.get("final_train_loss")
                or _latest_state_value(result, "latest_train_loss")
            ),
            "final_eval_loss": _number(
                metrics.get("eval_loss_mean")
                or metrics.get("best_eval_loss")
                or metrics.get("final_eval_loss")
                or _latest_state_value(result, "latest_eval_loss")
            ),
            "eval_loss_mean": _number(metrics.get("eval_loss_mean")),
            "pass_rate": _number(metrics.get("pass_rate")),
            "router_exact_match_rate": _number(
                metrics.get("router_exact_match_rate") or metrics.get("exact_match_rate")
            ),
            "finite": _bool_or_none(result.get("finite") if "finite" in result else metrics.get("finite")),
            "runtime_selected_device": _runtime_value(metrics, "selected_device"),
            "runtime_selected_precision": _runtime_value(metrics, "selected_precision"),
            "runtime_amp_enabled": _bool_or_none(_runtime_value(metrics, "amp_enabled")),
            "runtime_gpu_name": _runtime_value(metrics, "gpu_name"),
            "result_path": source,
            "error": result.get("error") or metrics.get("error"),
            "metadata": {
                "run_name": result.get("run_name"),
                "artifacts": result.get("artifacts"),
                "source_path": source,
                "model_ref": metrics.get("model_ref"),
                "dataset": metrics.get("dataset"),
                "dataset_ref": metrics.get("dataset_ref") or metrics.get("corpus_dataset_ref"),
                "dataset_split": metrics.get("dataset_split"),
                "dataset_version_id": metrics.get("dataset_version_id"),
            },
        }
    )
    _merge_parameter_fields(output, metrics)
    if output["routing"] is None:
        output["routing"] = _routing_from_model(output["model_type"])
    return _json_safe_row(output)


def _benchmark_row(
    metrics: dict[str, Any],
    source_id: str | None,
    *,
    component: str | None,
) -> dict[str, Any]:
    output = _empty_row()
    output.update(
        {
            "source_type": "benchmark",
            "source_id": source_id,
            "run_id": metrics.get("source_run_id") or metrics.get("run_id"),
            "kind": "benchmark",
            "mode": component or metrics.get("benchmark_type"),
            "model_type": metrics.get("model_type"),
            "routing": metrics.get("routing"),
            "trainable_policy_mode": metrics.get("trainable_policy_mode"),
            "use_fast_adapters": metrics.get("use_fast_adapters"),
            "use_generated_params": metrics.get("use_generated_params"),
            "finite": _bool_or_none(metrics.get("finite")),
            "runtime_selected_device": _runtime_value(metrics, "selected_device"),
            "runtime_selected_precision": _runtime_value(metrics, "selected_precision"),
            "runtime_amp_enabled": _bool_or_none(_runtime_value(metrics, "amp_enabled")),
            "runtime_gpu_name": _runtime_value(metrics, "gpu_name"),
            "result_path": metrics.get("metrics_path"),
            "error": metrics.get("error"),
            "metadata": {
                "benchmark_id": source_id,
                "benchmark_name": metrics.get("benchmark_name"),
                "benchmark_type": metrics.get("benchmark_type"),
                "benchmark_component": component,
                "checkpoint_path": metrics.get("checkpoint_path"),
                "model_ref": metrics.get("model_ref"),
                "dataset": metrics.get("dataset"),
                "status": metrics.get("status"),
            },
        }
    )
    _merge_metric_fields(output, metrics)
    return output


def _merge_metric_fields(output: dict[str, Any], metrics: dict[str, Any]) -> None:
    output["eval_loss_mean"] = _first_number(
        output.get("eval_loss_mean"),
        metrics.get("eval_loss_mean"),
    )
    output["final_eval_loss"] = _first_number(
        output.get("final_eval_loss"),
        metrics.get("final_eval_loss"),
        metrics.get("eval_loss_mean"),
        metrics.get("best_eval_loss"),
    )
    output["final_train_loss"] = _first_number(
        output.get("final_train_loss"),
        metrics.get("final_train_loss"),
        metrics.get("train_loss_last"),
    )
    output["pass_rate"] = _first_number(output.get("pass_rate"), metrics.get("pass_rate"))
    output["router_exact_match_rate"] = _first_number(
        output.get("router_exact_match_rate"),
        metrics.get("router_exact_match_rate"),
        metrics.get("exact_match_rate"),
    )
    _merge_parameter_fields(output, metrics)
    for key in ("model_type", "trainable_policy_mode", "use_fast_adapters", "use_generated_params"):
        if output.get(key) is None and metrics.get(key) is not None:
            output[key] = metrics.get(key)


def _merge_parameter_fields(output: dict[str, Any], values: dict[str, Any]) -> None:
    counts = values.get("parameter_counts") if isinstance(values, dict) else None
    total = values.get("total_params") if isinstance(values, dict) else None
    trainable = values.get("trainable_params") if isinstance(values, dict) else None
    frozen = values.get("frozen_params") if isinstance(values, dict) else None
    if isinstance(counts, dict):
        total = counts.get("total", total)
        trainable = counts.get("trainable", trainable)
        frozen = counts.get("frozen", frozen)
    output["total_params"] = _first_number(output.get("total_params"), total)
    output["trainable_params"] = _first_number(output.get("trainable_params"), trainable)
    output["frozen_params"] = _first_number(output.get("frozen_params"), frozen)
    output["trainable_ratio"] = _first_number(
        output.get("trainable_ratio"),
        values.get("trainable_ratio") if isinstance(values, dict) else None,
    )
    if output["trainable_ratio"] is None and output["total_params"]:
        output["trainable_ratio"] = float(output["trainable_params"] or 0) / float(output["total_params"])


def _result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        return dict(metrics)
    trainer_result = result.get("trainer_result")
    if isinstance(trainer_result, dict) and isinstance(trainer_result.get("metrics"), dict):
        return dict(trainer_result["metrics"])
    return dict(result)


def _result_config(result: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    for key in ("finetune_config", "continued_pretrain_config", "trainer_config", "config"):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if isinstance(value, dict):
            return dict(value)
        value = result.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _runtime_value(metrics: dict[str, Any], key: str) -> Any:
    runtime = metrics.get("runtime") if isinstance(metrics, dict) else None
    if isinstance(runtime, dict):
        return runtime.get(key)
    return metrics.get(f"runtime.{key}") if isinstance(metrics, dict) else None


def _kind_from_result(
    result: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
    source_path: str | None,
) -> str | None:
    source = str(source_path or "")
    if "continued_pretrain" in source or "continued_pretrain_config" in metrics:
        return "pretrain"
    if "trainer_result" in source and not (
        result.get("mode") or metrics.get("finetune_mode") or config.get("mode")
    ):
        return "trainer"
    if result.get("mode") or metrics.get("finetune_mode") or config.get("mode") or "finetune_result" in source:
        return "sft"
    if "trainer_result" in source or "final_state" in result:
        return "trainer"
    return result.get("kind") or metrics.get("kind")


def _latest_state_value(result: dict[str, Any], key: str):
    state = result.get("final_state")
    if isinstance(state, dict):
        return state.get(key)
    trainer = result.get("trainer_result")
    if isinstance(trainer, dict) and isinstance(trainer.get("final_state"), dict):
        return trainer["final_state"].get(key)
    return None


def _empty_row() -> dict[str, Any]:
    row = {key: None for key in NORMALIZED_KEYS}
    row["metadata"] = {}
    return row


def _number(value) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if isinstance(value, int) or result.is_integer():
        return int(result)
    return result


def _first_number(*values) -> int | float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _bool_or_none(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _routing_from_model(model_type) -> str | None:
    if model_type == "mop_oracle":
        return "oracle"
    if model_type == "mop_learned_router":
        return "learned_router"
    if model_type == "dense":
        return "none"
    return None


def _metadata(row: dict[str, Any], exclude: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in exclude}


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in row.items():
        if isinstance(value, dict):
            safe[key] = {
                str(child_key): json_safe(child_value)
                if not isinstance(child_value, (dict, list))
                else child_value
                for child_key, child_value in value.items()
                if child_value is not None
            }
        elif isinstance(value, list):
            safe[key] = value
        else:
            safe[key] = json_safe(value)
    return safe
