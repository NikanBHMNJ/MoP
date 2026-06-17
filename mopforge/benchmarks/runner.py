"""Benchmark runner and result schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopforge.benchmarks.config import BenchmarkConfig
from mopforge.benchmarks.evaluators import (
    evaluate_code_correctness,
    evaluate_composite,
    evaluate_loss,
    evaluate_parameter_efficiency,
    evaluate_router,
)
from mopforge.benchmarks.registry import BenchmarkRegistry, _now


@dataclass(slots=True)
class BenchmarkResult:
    """Result for one benchmark run."""

    benchmark_id: str
    name: str
    benchmark_type: str
    status: str
    metrics: dict[str, Any]
    metrics_path: str
    metrics_csv_path: str | None
    examples_path: str | None
    record_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "benchmark_type": self.benchmark_type,
            "status": self.status,
            "metrics": dict(self.metrics),
            "metrics_path": self.metrics_path,
            "metrics_csv_path": self.metrics_csv_path,
            "examples_path": self.examples_path,
            "record_path": self.record_path,
        }


def run_benchmark(
    config: BenchmarkConfig,
    registry_root: str | Path = "benchmarks",
) -> BenchmarkResult:
    """Run one local CPU benchmark and persist metrics/examples."""

    config = BenchmarkConfig.from_dict(config.to_dict())
    root = config.output_root if str(registry_root) == "benchmarks" else registry_root
    registry = BenchmarkRegistry(root)
    record = registry.create_benchmark(config)
    record.status = "running"
    registry.save_record(record)
    examples_path = None
    try:
        metrics = _dispatch(config)
        status = "completed"
    except Exception as exc:
        metrics = {
            "benchmark_type": config.benchmark_type,
            "error": str(exc),
            "failed": True,
            "source_run_id": config.run_id,
            "checkpoint_path": config.checkpoint_path,
        }
        status = "failed"
    dataset_metadata = _dataset_metadata(config)
    if dataset_metadata:
        metrics["dataset"] = dataset_metadata
    if config.model_ref:
        metrics["model_ref"] = config.model_ref
    _attach_runtime_flattened(metrics)
    examples = metrics.pop("examples", None)
    if examples is not None and not isinstance(examples, list):
        metrics["examples"] = examples
        examples = None
    metrics["benchmark_id"] = record.benchmark_id
    metrics["benchmark_name"] = config.name
    metrics["status"] = status
    if examples is not None:
        examples_path = registry.write_examples(record.benchmark_id, list(examples))
    metrics_path = registry.write_metrics(record.benchmark_id, metrics)
    metrics_csv_path = registry.write_metrics_csv(record.benchmark_id, [metrics])

    record.status = status
    record.completed_at = _now()
    record.metrics_path = str(metrics_path)
    record.metrics_csv_path = str(metrics_csv_path)
    record.examples_path = str(examples_path) if examples_path is not None else None
    record.metadata.update(
        {
            "source_run_id": config.run_id,
            "checkpoint_path": config.checkpoint_path,
            "model_ref": config.model_ref,
            "dataset_ref": config.dataset_ref,
            "dataset_split": config.dataset_split,
            "dataset": dataset_metadata,
        }
    )
    registry.save_record(record)
    record_path = registry.benchmark_dir(record.benchmark_id) / "record.json"
    return BenchmarkResult(
        benchmark_id=record.benchmark_id,
        name=config.name,
        benchmark_type=config.benchmark_type,
        status=status,
        metrics=metrics,
        metrics_path=str(metrics_path),
        metrics_csv_path=str(metrics_csv_path),
        examples_path=str(examples_path) if examples_path is not None else None,
        record_path=str(record_path),
    )


def _dispatch(config: BenchmarkConfig) -> dict[str, Any]:
    if config.benchmark_type == "loss":
        return evaluate_loss(config)
    if config.benchmark_type == "code_correctness":
        return evaluate_code_correctness(config)
    if config.benchmark_type == "router":
        return evaluate_router(config)
    if config.benchmark_type == "parameter_efficiency":
        return evaluate_parameter_efficiency(config)
    if config.benchmark_type == "composite":
        return evaluate_composite(config)
    raise ValueError(f"Unsupported benchmark_type: {config.benchmark_type}")


def _dataset_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    if not config.dataset_ref:
        return {}
    try:
        from mopforge.datasets import DatasetRegistry

        manifest = DatasetRegistry().resolve_dataset_ref(config.dataset_ref)
        return {
            "dataset_ref": config.dataset_ref,
            "dataset_id": manifest.dataset_id,
            "version_id": manifest.version_id,
            "kind": manifest.kind,
            "split": config.dataset_split,
            "combined_sha256": manifest.combined_sha256,
            "record_count": manifest.stats.record_count,
        }
    except Exception as exc:
        return {
            "dataset_ref": config.dataset_ref,
            "split": config.dataset_split,
            "error": str(exc),
        }


def _attach_runtime_flattened(metrics: dict[str, Any]) -> None:
    runtime = metrics.get("runtime")
    if not isinstance(runtime, dict):
        return
    for source_key, output_key in (
        ("requested_device", "runtime.requested_device"),
        ("selected_device", "runtime.selected_device"),
        ("requested_precision", "runtime.requested_precision"),
        ("selected_precision", "runtime.selected_precision"),
        ("amp_enabled", "runtime.amp_enabled"),
        ("cuda_available", "runtime.cuda_available"),
        ("gpu_name", "runtime.gpu_name"),
    ):
        metrics[output_key] = runtime.get(source_key)
