"""Sequential local CPU experiment matrix runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopforge.experiments.matrix import ExperimentConfig, expand_experiment_matrix
from mopforge.experiments.registry import ExperimentRegistry


@dataclass(slots=True)
class ExperimentRunResult:
    """Result summary for one local experiment run."""

    experiment_id: str
    status: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    run_records: list[dict[str, Any]]
    summary_path: str
    summary_csv_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "total_runs": self.total_runs,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "run_records": [dict(record) for record in self.run_records],
            "summary_path": self.summary_path,
            "summary_csv_path": self.summary_csv_path,
        }


def run_experiment(
    config: ExperimentConfig,
    registry_root: str | Path = "experiments",
) -> ExperimentRunResult:
    """Run one local experiment config sequentially on CPU."""

    config = ExperimentConfig.from_dict(config.to_dict())
    runs = expand_experiment_matrix(config)
    registry = ExperimentRegistry(registry_root)
    record = registry.create_experiment(config)
    record.status = "running"
    record.total_runs = len(runs)
    registry.save_record(record)
    expanded_path = registry.write_expanded_runs(record.experiment_id, runs)

    run_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    run_ids: list[str] = []

    for index, envelope in enumerate(runs):
        run_record: dict[str, Any] = {
            "experiment_id": record.experiment_id,
            "index": index,
            "kind": envelope.kind,
            "config": envelope.to_dict(),
            "status": "pending",
        }
        try:
            child_result = _run_envelope(envelope)
            run_id = _result_run_id(child_result)
            result_path = _result_path(child_result)
            metrics = _result_metrics(child_result)
            artifacts = _result_artifacts(child_result)
            run_record.update(
                {
                    "status": "completed",
                    "run_id": run_id,
                    "result_path": result_path,
                    "metrics": metrics,
                    "artifacts": artifacts,
                }
            )
            completed += 1
            if run_id:
                run_ids.append(run_id)
        except Exception as exc:
            run_record.update({"status": "failed", "error": str(exc)})
            failed += 1
        summary_rows.append(_summary_row(record.experiment_id, index, envelope, run_record))
        registry.write_run_record(record.experiment_id, index, run_record)
        run_records.append(run_record)

    status = _final_status(completed, failed, len(runs))
    summary = {
        "experiment_id": record.experiment_id,
        "name": config.name,
        "status": status,
        "total_runs": len(runs),
        "completed_runs": completed,
        "failed_runs": failed,
        "run_ids": list(run_ids),
        "expanded_runs_path": str(expanded_path),
        "rows": summary_rows,
    }
    summary_path = registry.write_summary(record.experiment_id, summary)
    summary_csv_path = registry.write_summary_csv(record.experiment_id, summary_rows)

    record.status = status
    record.total_runs = len(runs)
    record.completed_runs = completed
    record.failed_runs = failed
    record.run_ids = list(run_ids)
    record.summary_path = str(summary_path)
    record.metadata["summary_csv_path"] = str(summary_csv_path)
    record.metadata["expanded_runs_path"] = str(expanded_path)
    registry.save_record(record)

    return ExperimentRunResult(
        experiment_id=record.experiment_id,
        status=status,
        total_runs=len(runs),
        completed_runs=completed,
        failed_runs=failed,
        run_records=run_records,
        summary_path=str(summary_path),
        summary_csv_path=str(summary_csv_path),
    )


def _run_envelope(envelope):
    from mopforge.configs.validation import (
        finetune_config_from_envelope,
        pretrain_config_from_envelope,
        trainer_config_from_envelope,
    )
    from mopforge.pretrain import run_continued_pretraining
    from mopforge.sft import run_finetune
    from mopforge.training import TinyTrainer

    if envelope.kind == "sft":
        return run_finetune(finetune_config_from_envelope(envelope))
    if envelope.kind == "pretrain":
        return run_continued_pretraining(pretrain_config_from_envelope(envelope))
    if envelope.kind == "trainer":
        return TinyTrainer(trainer_config_from_envelope(envelope)).train()
    raise ValueError(f"Unsupported experiment child kind: {envelope.kind}")


def _summary_row(
    experiment_id: str,
    index: int,
    envelope,
    run_record: dict[str, Any],
) -> dict[str, Any]:
    metrics = dict(run_record.get("metrics") or {})
    payload = dict(envelope.payload)
    trainable_policy = metrics.get("trainable_policy")
    if isinstance(trainable_policy, dict):
        policy_mode = trainable_policy.get("mode")
    else:
        policy_mode = payload.get("trainable_policy_mode")
    return {
        "experiment_id": experiment_id,
        "index": index,
        "kind": envelope.kind,
        "status": run_record.get("status"),
        "run_id": run_record.get("run_id"),
        "mode": payload.get("mode"),
        "model_type": payload.get("model_type") or metrics.get("model_type"),
        "trainable_policy_mode": policy_mode,
        "final_train_loss": _first_present(
            metrics,
            "train_loss_last",
            "final_train_loss",
        ),
        "final_eval_loss": _first_present(
            metrics,
            "eval_loss_mean",
            "final_eval_loss",
        ),
        "finite": metrics.get("finite"),
        "result_path": run_record.get("result_path"),
        "model_ref": payload.get("model_ref"),
        "dataset_ref": payload.get("dataset_ref") or payload.get("corpus_dataset_ref"),
        "dataset_split": payload.get("dataset_split"),
        "dataset_version_id": payload.get("dataset_version_id"),
        "runtime_selected_device": _runtime_value(metrics, "selected_device"),
        "runtime_selected_precision": _runtime_value(metrics, "selected_precision"),
        "runtime_amp_enabled": _runtime_value(metrics, "amp_enabled"),
        "runtime_gpu_name": _runtime_value(metrics, "gpu_name"),
        "error": run_record.get("error"),
    }


def _result_run_id(result) -> str | None:
    return getattr(result, "run_id", None)


def _result_path(result) -> str | None:
    artifacts = _result_artifacts(result)
    for key in (
        "finetune_result_json",
        "continued_pretrain_result_json",
        "trainer_result_json",
    ):
        if artifacts.get(key):
            return str(artifacts[key])
    return None


def _result_metrics(result) -> dict[str, Any]:
    return dict(getattr(result, "metrics", {}) or {})


def _result_artifacts(result) -> dict[str, Any]:
    return dict(getattr(result, "artifacts", {}) or {})


def _first_present(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def _runtime_value(metrics: dict[str, Any], key: str) -> Any:
    runtime = metrics.get("runtime")
    if isinstance(runtime, dict):
        return runtime.get(key)
    return metrics.get(f"runtime.{key}")


def _final_status(completed: int, failed: int, total: int) -> str:
    if total == 0 or completed == 0:
        return "failed"
    if failed:
        return "completed_with_failures"
    return "completed"
