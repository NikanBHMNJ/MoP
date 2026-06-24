"""Staged single-device feasibility probe for large GPU training profiles."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from mopforge.gpu.checkpointing import load_gpu_checkpoint, restore_gpu_checkpoint
from mopforge.gpu.config import GPUTrainingConfig
from mopforge.gpu.memory import (
    cuda_memory_metrics,
    estimate_from_config,
    reset_cuda_peak_memory,
    system_memory_metrics,
)
from mopforge.gpu.trainer import GPUTrainer
from mopforge.runtime import move_batch_to_device


def run_gpu_probe(
    config: GPUTrainingConfig,
    *,
    optimizer_updates: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a staged allocation/training/resume probe and write a JSON report."""

    update_budget = int(
        optimizer_updates
        if optimizer_updates is not None
        else config.metadata.get("probe_optimizer_steps", 20)
    )
    if update_budget <= 0:
        raise ValueError("optimizer_updates must be positive.")

    payload = config.to_dict()
    payload.update(
        {
            "max_optimizer_steps": update_budget,
            "save_full_checkpoints": False,
            "save_optimizer_state": False,
            "run_generation_eval": False,
        }
    )
    probe_config = GPUTrainingConfig.from_dict(payload)
    trainer = GPUTrainer(probe_config)
    phases: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "report_format": "mopforge_gpu_feasibility_probe_v1",
        "status": "running",
        "config": probe_config.to_dict(),
        "requested_optimizer_updates": update_budget,
        "static_memory_estimate": estimate_from_config(probe_config).to_dict(),
        "phases": phases,
        "automatic_config_changes": [],
    }
    report_path = Path(output_path) if output_path else None

    def phase(name: str, operation: Callable[[], Any]) -> Any:
        _synchronize(trainer)
        reset_cuda_peak_memory(trainer.runtime)
        before = _telemetry(trainer, report_path or trainer.output_dir)
        started = time.perf_counter()
        try:
            value = operation()
        except Exception as exc:
            _synchronize(trainer)
            phases.append(
                {
                    "name": name,
                    "status": "oom" if _is_oom(exc) else "failed",
                    "duration_sec": round(time.perf_counter() - started, 6),
                    "before": before,
                    "after": _telemetry(trainer, report_path or trainer.output_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        _synchronize(trainer)
        after = _telemetry(trainer, report_path or trainer.output_dir)
        item = {
            "name": name,
            "status": "completed",
            "duration_sec": round(time.perf_counter() - started, 6),
            "before": before,
            "after": after,
        }
        if isinstance(value, dict) and "phase_metrics" in value:
            item.update(dict(value["phase_metrics"]))
        phases.append(item)
        return value

    try:
        _validate_probe_hardware(probe_config)
        phase("model_and_data_allocation", lambda: trainer.setup(initialize_optimizer=False))
        report["runtime"] = dict(trainer.runtime_meta)
        report["actual_model_storage"] = _model_storage(trainer.model)
        report["memory_estimate_reconciliation"] = {
            "estimated_parameter_count": report["static_memory_estimate"]["parameter_count"],
            "actual_parameter_count": report["actual_model_storage"]["parameter_count"],
            "estimated_trainable_parameter_count": report["static_memory_estimate"][
                "trainable_parameter_count"
            ],
            "actual_trainable_parameter_count": report["actual_model_storage"][
                "trainable_parameter_count"
            ],
            "actual_parameter_storage_gb": report["actual_model_storage"]["storage_gb"],
        }

        batch = move_batch_to_device(
            trainer._next_train_batch(),
            trainer.runtime.device_info.selected,
        )
        forward: dict[str, Any] = {}

        def forward_phase() -> dict[str, Any]:
            trainer.model.train()
            loss = trainer._forward_loss(batch, include_distillation=True)
            forward["loss"] = loss
            value = float(loss.detach().float().cpu().item())
            forward["loss_value"] = value
            return {
                "phase_metrics": {
                    "loss": value,
                    "tokens": _token_count(batch),
                }
            }

        phase("forward", forward_phase)

        def backward_phase() -> dict[str, Any]:
            forward["loss"].div(float(probe_config.gradient_accumulation_steps)).backward()
            return {"phase_metrics": {"loss": forward["loss_value"]}}

        phase("backward", backward_phase)
        for parameter in trainer.model.parameters():
            parameter.grad = None
        phase("optimizer_object_allocation", trainer.initialize_optimizer)

        update_losses: list[float] = []
        update_tokens: list[int] = []
        update_durations: list[float] = []

        def optimizer_updates_phase() -> dict[str, Any]:
            started = time.perf_counter()
            for _ in range(update_budget):
                update_started = time.perf_counter()
                loss_value, tokens = _run_optimizer_update(trainer)
                update_losses.append(loss_value)
                update_tokens.append(tokens)
                update_durations.append(time.perf_counter() - update_started)
            elapsed = max(time.perf_counter() - started, 1e-9)
            return {
                "phase_metrics": {
                    "optimizer_updates": update_budget,
                    "first_loss": update_losses[0],
                    "final_loss": update_losses[-1],
                    "mean_loss": sum(update_losses) / len(update_losses),
                    "tokens": sum(update_tokens),
                    "tokens_per_sec": round(sum(update_tokens) / elapsed, 4),
                }
            }

        phase("optimizer_state_and_steady_updates", optimizer_updates_phase)
        eval_result = phase("evaluation", trainer.evaluate)
        report["evaluation"] = dict(eval_result)

        checkpoint: dict[str, Any] = {}

        def save_phase() -> dict[str, Any]:
            path = trainer.save_checkpoint(
                trainer.state.global_step,
                tag="probe-model-only",
                record_latest=False,
                model_only=True,
            )
            checkpoint["path"] = path
            checkpoint["size_bytes"] = Path(path).stat().st_size
            checkpoint["loss_before"] = _eval_batch_loss(trainer, batch)
            checkpoint["parameter"] = _first_trainable_parameter(trainer.model)
            checkpoint["parameter_value"] = _parameter_sample(checkpoint["parameter"])
            return {
                "phase_metrics": {
                    "checkpoint_path": path,
                    "checkpoint_size_bytes": checkpoint["size_bytes"],
                    "optimizer_state_saved": False,
                }
            }

        phase("atomic_model_only_checkpoint_save", save_phase)

        def load_phase() -> dict[str, Any]:
            parameter = checkpoint["parameter"]
            with _require_torch().no_grad():
                parameter.zero_()
            loaded = load_gpu_checkpoint(
                checkpoint["path"],
                map_location="cpu",
            )
            metadata = restore_gpu_checkpoint(
                loaded,
                model=trainer.model,
                restore_rng=False,
                restore_optimizer=False,
                restore_scheduler=False,
                restore_scaler=False,
                strict_model=not probe_config.save_trainable_only_checkpoints,
            )
            restored_value = _parameter_sample(parameter)
            loss_after = _eval_batch_loss(trainer, batch)
            loss_delta = abs(float(loss_after) - float(checkpoint["loss_before"]))
            value_delta = abs(restored_value - float(checkpoint["parameter_value"]))
            checkpoint.update(
                {
                    "loss_after": loss_after,
                    "loss_delta": loss_delta,
                    "parameter_value_delta": value_delta,
                    "restore_metadata": metadata,
                    "passed": loss_delta <= 1e-5 and value_delta <= 1e-6,
                }
            )
            return {
                "phase_metrics": {
                    "loss_before": checkpoint["loss_before"],
                    "loss_after": loss_after,
                    "loss_delta": loss_delta,
                    "parameter_value_delta": value_delta,
                    "resume_probe_passed": checkpoint["passed"],
                }
            }

        phase("checkpoint_load_and_resume_consistency", load_phase)
        report["checkpoint_resume_probe"] = {
            key: value for key, value in checkpoint.items() if key != "parameter"
        }

        mean_update_sec = sum(update_durations) / max(1, len(update_durations))
        report["training_probe"] = {
            "optimizer_updates": update_budget,
            "microsteps": trainer.state.global_step,
            "losses": update_losses,
            "finite_loss": all(math.isfinite(value) for value in update_losses),
            "loss_decreased": update_losses[-1] < update_losses[0],
            "mean_optimizer_update_sec": round(mean_update_sec, 6),
            "tokens_per_sec": round(
                sum(update_tokens) / max(sum(update_durations), 1e-9),
                4,
            ),
        }
        report["runtime_projection"] = {
            str(updates): {
                "optimizer_updates": updates,
                "seconds": round(mean_update_sec * updates, 2),
                "hours": round(mean_update_sec * updates / 3600.0, 3),
            }
            for updates in (500, 2000)
        }
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "oom" if _is_oom(exc) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        _cleanup(trainer)
        phases.append(
            {
                "name": "cleanup",
                "status": "completed",
                "after": _telemetry(trainer, report_path or trainer.output_dir),
            }
        )
        trainer.close()

    report["acceptance"] = _acceptance(report)
    if report_path is None:
        report_path = trainer.output_dir / "gpu_probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _run_optimizer_update(trainer: GPUTrainer) -> tuple[float, int]:
    torch = _require_torch()
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    tokens = 0
    for _ in range(trainer.config.gradient_accumulation_steps):
        batch = move_batch_to_device(
            trainer._next_train_batch(),
            trainer.runtime.device_info.selected,
        )
        loss = trainer._forward_loss(batch, include_distillation=True)
        trainer.scaler.scale(
            loss / float(trainer.config.gradient_accumulation_steps)
        ).backward()
        losses.append(float(loss.detach().float().cpu().item()))
        batch_tokens = _token_count(batch)
        tokens += batch_tokens
        trainer.state.global_step += 1
        trainer.state.samples_seen += _batch_size(batch)
        trainer.state.tokens_seen += batch_tokens
    if trainer.config.max_grad_norm is not None:
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in trainer.model.parameters() if parameter.requires_grad],
            trainer.config.max_grad_norm,
        )
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.state.optimizer_step += 1
    if trainer.scheduler is not None:
        trainer.scheduler.step()
    trainer.state.latest_train_loss = losses[-1]
    return sum(losses) / len(losses), tokens


def _eval_batch_loss(trainer: GPUTrainer, batch: dict[str, Any]) -> float:
    torch = _require_torch()
    previous = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            loss = trainer._forward_loss(batch, include_distillation=False)
        return float(loss.detach().float().cpu().item())
    finally:
        trainer.model.train(previous)


def _model_storage(model) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    total_params = 0
    total_bytes = 0
    trainable_params = 0
    for parameter in model.parameters():
        key = str(parameter.dtype).replace("torch.", "")
        item = counts.setdefault(key, {"parameters": 0, "bytes": 0})
        count = int(parameter.numel())
        size = count * int(parameter.element_size())
        item["parameters"] += count
        item["bytes"] += size
        total_params += count
        total_bytes += size
        if parameter.requires_grad:
            trainable_params += count
    return {
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "storage_bytes": total_bytes,
        "storage_gb": round(total_bytes / (1024**3), 4),
        "by_dtype": counts,
    }


def _acceptance(report: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(report.get("runtime") or {})
    total_gb = runtime.get("gpu_memory_gb")
    if total_gb is None:
        for phase in report.get("phases", []):
            total_gb = (phase.get("after") or {}).get("cuda", {}).get("device_total_gb")
            if total_gb is not None:
                break
    reserved_values = [
        (phase.get("after") or {}).get("cuda", {}).get("peak_reserved_gb")
        for phase in report.get("phases", [])
    ]
    reserved_values = [float(value) for value in reserved_values if value is not None]
    peak_reserved = max(reserved_values) if reserved_values else None
    configured_threshold = (
        (report.get("config") or {}).get("metadata") or {}
    ).get("probe_peak_reserved_limit_gb")
    threshold = (
        float(configured_threshold)
        if configured_threshold is not None
        else (34.0 if total_gb is not None and float(total_gb) < 60.0 else 68.0)
    )
    training = dict(report.get("training_probe") or {})
    checkpoint = dict(report.get("checkpoint_resume_probe") or {})
    gates = {
        "probe_completed": report.get("status") == "completed",
        "finite_loss": training.get("finite_loss") is True,
        "loss_decreased": training.get("loss_decreased") is True,
        "checkpoint_resume_passed": checkpoint.get("passed") is True,
        "peak_reserved_within_limit": (
            peak_reserved is not None and peak_reserved <= threshold
        ),
        "no_oom": report.get("status") != "oom",
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "peak_reserved_gb": peak_reserved,
        "peak_reserved_limit_gb": threshold,
        "detected_total_gpu_memory_gb": total_gb,
    }


def _telemetry(trainer: GPUTrainer, path: str | Path) -> dict[str, Any]:
    return {
        "cuda": cuda_memory_metrics(trainer.runtime),
        "system": system_memory_metrics(path),
    }


def _cleanup(trainer: GPUTrainer) -> None:
    try:
        trainer.optimizer = None
        trainer.scheduler = None
        if trainer.model is not None:
            trainer.model.to("cpu")
        trainer._empty_cache()
    except Exception:
        pass


def _synchronize(trainer: GPUTrainer) -> None:
    try:
        torch = _require_torch()
        if trainer.runtime is not None and trainer.runtime.device_info.device_type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _first_trainable_parameter(model):
    for parameter in model.parameters():
        if parameter.requires_grad:
            return parameter
    raise RuntimeError("Probe model has no trainable parameter.")


def _parameter_sample(parameter) -> float:
    return float(parameter.detach().reshape(-1)[0].float().cpu().item())


def _batch_size(batch: dict[str, Any]) -> int:
    value = batch.get("input_ids", batch.get("hidden_states"))
    return int(value.shape[0])


def _token_count(batch: dict[str, Any]) -> int:
    mask = batch.get("attention_mask")
    if mask is not None:
        return int(mask.sum().detach().cpu().item())
    value = batch.get("input_ids", batch.get("hidden_states"))
    return int(value.shape[0] * value.shape[1])


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: memory allocation" in text


def _validate_probe_hardware(config: GPUTrainingConfig) -> None:
    torch = _require_torch()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested but CUDA is unavailable; CPU fallback is disabled for probes.")
    required_name = config.metadata.get("required_gpu_name_contains")
    if required_name and config.device == "cuda":
        detected = torch.cuda.get_device_name(0)
        if str(required_name).lower() not in detected.lower():
            raise RuntimeError(
                f"Probe requires GPU name containing {required_name!r}, detected {detected!r}."
            )
    if config.device == "cuda" and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        minimum = config.metadata.get("minimum_gpu_memory_gb")
        maximum = config.metadata.get("maximum_gpu_memory_gb")
        if minimum is not None and total_gb < float(minimum):
            raise RuntimeError(
                f"Probe requires at least {minimum} GiB, detected {total_gb:.2f} GiB."
            )
        if maximum is not None and total_gb > float(maximum):
            raise RuntimeError(
                f"Probe requires at most {maximum} GiB, detected {total_gb:.2f} GiB."
            )


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GPU feasibility probes.") from exc
    return torch
