"""Single-device GPU-aware trainer with CPU fallback."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.artifacts import ArtifactManager, ArtifactRecord
from mopforge.gpu.checkpointing import (
    load_gpu_checkpoint,
    restore_gpu_checkpoint,
    save_gpu_checkpoint,
)
from mopforge.gpu.activation_cache import (
    build_cached_activation_dataloaders,
)
from mopforge.gpu.config import GPUTrainingConfig, GPUTrainingResult, GPUTrainingState
from mopforge.gpu.data import GPUDataConfig, build_gpu_dataloaders, load_gpu_lesson_splits
from mopforge.gpu.memory import (
    cuda_memory_metrics,
    estimate_from_config,
    reset_cuda_peak_memory,
    write_memory_estimate,
)
from mopforge.gpu.mop_execution import estimate_active_parameters, fast_parameter_metadata
from mopforge.gpu.registry import GPURunRecord, GPURunRegistry
from mopforge.gpu.scaler import AmpScaler
from mopforge.models import (
    ModelArchitectureConfig,
    ModelRegistry,
    adapter_names_from_target_modules,
    build_tiny_model_from_architecture,
    condition_names_from_target_modules,
)
from mopforge.eval import evaluate_generated_code_for_lesson, summarize_generation_results
from mopforge.runtime import (
    RuntimeConfig,
    apply_runtime_determinism,
    autocast_context,
    build_runtime_context,
    move_batch_to_device,
    move_model_to_runtime,
    runtime_metadata,
)
from mopforge.tokenization import TokenizerSpec, build_tokenizer
from mopforge.training.parameter_policy import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    build_optimizer_for_trainable_parameters,
    count_parameters,
)


class GPUTrainer:
    """A serious single-device beta trainer that remains CPU-testable."""

    def __init__(self, config: GPUTrainingConfig) -> None:
        self.config = config
        self.state = GPUTrainingState()
        self.run_id = config.run_id or _make_run_id(config.name)
        self.registry = GPURunRegistry(config.output_root)
        self.output_dir = self.registry.run_dir(self.run_id)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.artifact_manager = ArtifactManager(config.artifact_root)
        self.runtime = None
        self.runtime_meta: dict[str, Any] = {}
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler: AmpScaler | None = None
        self.train_loader = None
        self.eval_loader = None
        self._train_iter = None
        self._data_config: GPUDataConfig | None = None
        self.data_metadata: dict[str, Any] = {}
        self.model_metadata: dict[str, Any] = {}
        self.checkpoint_metadata: dict[str, Any] = {}
        self.artifacts: dict[str, Any] = {}
        self._trainable_policy: TrainableParameterPolicy | None = None
        self._train_started_at: float | None = None
        self._train_finished_at: float | None = None
        self._step_times: list[float] = []
        self._setup_done = False

    def setup(self) -> None:
        """Build runtime, model, data, optimizer, and optional resume state."""

        torch = _require_torch()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.runtime = build_runtime_context(_runtime_config(self.config))
        apply_runtime_determinism(self.runtime, self.config.metadata.get("seed"))
        self.runtime_meta = runtime_metadata(self.runtime)
        self.state.runtime_metadata = dict(self.runtime_meta)
        self.scaler = AmpScaler(self.runtime)

        self.tokenizer = build_tokenizer(TokenizerSpec(tokenizer_type="byte"))
        self.model = self._build_model()
        self.model = move_model_to_runtime(self.model, self.runtime)
        activation_metadata = apply_activation_checkpointing(
            self.model,
            self.config.activation_checkpointing,
        )
        attention_metadata = select_attention_metadata(self.config.efficient_attention)
        policy = TrainableParameterPolicy(
            mode=self.config.trainable_policy_mode,
            target_modules=self.config.target_modules or None,
            train_router=bool(self.config.metadata.get("train_router", False)),
            train_lm_head=bool(
                self.config.metadata.get("train_lm_head", False)
                or self.config.trainable_policy_mode == "adapters_norm_head"
            ),
            train_norm=bool(
                self.config.metadata.get("train_norm", False)
                or self.config.trainable_policy_mode == "adapters_norm_head"
            ),
            train_fast_adapters=self.config.use_fast_adapters,
            train_lora_deltas=self.config.use_lora_deltas,
            train_generated_params=self.config.use_generated_params,
            metadata={"training_kind": "gpu_train"},
        )
        self._trainable_policy = policy
        group_summaries = apply_trainable_policy(self.model, policy)
        params = count_parameters(self.model)
        self.model_metadata.update(
            {
                "parameter_counts": params,
                "parameter_group_summaries": [item.to_dict() for item in group_summaries],
                "trainable_param_ratio": params["trainable"] / max(1, params["total"]),
                "activation_checkpointing": activation_metadata,
                "efficient_attention": attention_metadata,
            }
        )
        self.optimizer = build_optimizer_for_trainable_parameters(
            self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = _build_scheduler(self.optimizer, self.config)
        data_config = GPUDataConfig(
            dataset_ref=self.config.dataset_ref,
            dataset_split=self.config.dataset_split,
            dataset_split_id=self.config.dataset_split_id,
            lesson_path=self.config.lesson_path,
            corpus_path=self.config.corpus_path,
            max_seq_len=self.config.max_seq_len,
            micro_batch_size=self.config.micro_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            streaming=bool(self.config.metadata.get("streaming", False)),
            seed=int(self.config.metadata.get("seed", 42)),
            max_examples=_max_examples(self.config),
        )
        self._data_config = data_config
        if self.config.activation_cache_path:
            pin_memory = bool(self.config.pin_memory and self.runtime.device_info.device_type == "cuda")
            self.train_loader, self.eval_loader, self.data_metadata = build_cached_activation_dataloaders(
                self.config.activation_cache_path,
                micro_batch_size=self.config.micro_batch_size,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
            )
        else:
            self.train_loader, self.eval_loader, self.data_metadata = build_gpu_dataloaders(
                data_config,
                self.tokenizer,
                self.runtime,
            )
        self._train_iter = cycle(self.train_loader)

        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        self._write_setup_artifacts()
        self._setup_done = True

    def train(self) -> GPUTrainingResult:
        if not self._setup_done:
            self.setup()
        torch = _require_torch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        status = "completed"
        reset_cuda_peak_memory(self.runtime)
        self._train_started_at = time.perf_counter()
        try:
            while self.state.global_step < self.config.max_steps:
                step_started_at = time.perf_counter()
                batch = move_batch_to_device(next(self._train_iter), self.runtime.device_info.selected)
                loss = self._forward_loss(batch)
                scaled_loss = loss / float(self.config.gradient_accumulation_steps)
                self.scaler.scale(scaled_loss).backward()
                self.state.global_step += 1
                self.state.samples_seen += _batch_size(batch)
                self.state.tokens_seen += _token_count(batch)
                self.state.latest_train_loss = float(loss.detach().float().cpu().item())
                should_step = (
                    self.state.global_step % self.config.gradient_accumulation_steps == 0
                    or self.state.global_step >= self.config.max_steps
                )
                if should_step:
                    if self.config.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.optimizer_step += 1
                    if self.scheduler is not None:
                        self.scheduler.step()
                if self.state.global_step % self.config.eval_every_steps == 0:
                    eval_metrics = self.evaluate()
                    self.state.latest_eval_loss = eval_metrics.get("eval_loss_mean")
                    improved = self.state.best_eval_loss is None or (
                        self.state.latest_eval_loss is not None
                        and self.state.latest_eval_loss
                        < self.state.best_eval_loss - self.config.early_stopping_min_delta
                    )
                    if improved:
                        self.state.best_eval_loss = self.state.latest_eval_loss
                        self.state.evals_without_improvement = 0
                    else:
                        self.state.evals_without_improvement += 1
                    if (
                        self.config.early_stopping_enabled
                        and self.state.evals_without_improvement
                        >= self.config.early_stopping_patience_evals
                    ):
                        self.state.stopped_early = True
                        self.state.stop_reason = "eval_loss_patience_exhausted"
                        break
                if self.state.global_step % self.config.log_every_steps == 0:
                    self._append_metrics({"event": "log"})
                if self.config.save_full_checkpoints and self.state.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint(self.state.global_step)
                if self.config.empty_cache_every_steps and self.state.global_step % self.config.empty_cache_every_steps == 0:
                    self._empty_cache()
                self._step_times.append(time.perf_counter() - step_started_at)
            if self.config.save_full_checkpoints and not self.state.latest_checkpoint_path:
                self.save_checkpoint(self.state.global_step)
        except Exception:
            status = "failed"
            raise
        finally:
            self._train_finished_at = time.perf_counter()
            self._write_state()
        return self._finish(status)

    def evaluate(self) -> dict[str, Any]:
        if not self._setup_done:
            self.setup()
        losses: list[float] = []
        self.model.eval()
        try:
            import torch

            with torch.no_grad():
                for index, batch in enumerate(self.eval_loader):
                    if index >= self.config.eval_batches:
                        break
                    batch = move_batch_to_device(batch, self.runtime.device_info.selected)
                    loss = self._forward_loss(batch)
                    losses.append(float(loss.detach().float().cpu().item()))
        finally:
            self.model.train()
        mean_loss = sum(losses) / len(losses) if losses else None
        metrics = {
            "eval_loss_mean": mean_loss,
            "eval_batches": len(losses),
            "step": self.state.global_step,
        }
        self._append_metrics(metrics)
        return metrics

    def save_checkpoint(self, step: int) -> str:
        path = self.checkpoint_dir / f"checkpoint-step-{step:06d}.pt"
        base_checkpoint_path = None
        if self.config.resume_model_only:
            base_checkpoint_path = self.checkpoint_metadata.get("checkpoint_path")
        if not base_checkpoint_path:
            base_checkpoint_path = self.config.base_checkpoint_path
        if not base_checkpoint_path and self.config.resume_model_only:
            base_checkpoint_path = self.config.resume_from_checkpoint
        saved = save_gpu_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer if self.config.save_optimizer_state else None,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=self.state,
            config=self.config,
            runtime_metadata=self.runtime_meta,
            data_metadata=self.data_metadata,
            model_metadata=self.model_metadata,
            memory_metadata=self._memory_snapshot(),
            trainable_only=self.config.save_trainable_only_checkpoints,
            base_checkpoint_path=base_checkpoint_path,
            trainable_policy=(
                self._trainable_policy.to_dict()
                if self._trainable_policy is not None
                else None
            ),
        )
        self.state.latest_checkpoint_path = saved
        self.checkpoint_metadata = {
            "latest_checkpoint_path": saved,
            "trainable_only": self.config.save_trainable_only_checkpoints,
        }
        self._register_artifact(
            ArtifactRecord(
                artifact_id=f"gpu-checkpoint-{self.run_id}-{step}-{uuid4().hex[:8]}",
                kind="checkpoint",
                path=saved,
                run_id=self.run_id,
                model_type=self.config.model_type,
                step=step,
                metadata={
                    "training_kind": "gpu_train",
                    "global_step": step,
                    "optimizer_step": self.state.optimizer_step,
                    "tokens_seen": self.state.tokens_seen,
                    "runtime": dict(self.runtime_meta),
                    "trainable_only": self.config.save_trainable_only_checkpoints,
                },
            )
        )
        return saved

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        checkpoint_path = _resolve_checkpoint_reference(path, self.config.output_root)
        payload = load_gpu_checkpoint(checkpoint_path, map_location=self.runtime.device_info.selected if self.runtime else "cpu")
        model_only = bool(self.config.resume_model_only)
        metadata = restore_gpu_checkpoint(
            payload,
            model=self.model,
            optimizer=None if model_only else self.optimizer,
            scheduler=None if model_only else self.scheduler,
            scaler=None if model_only else self.scaler,
            restore_rng=False if model_only else self.config.save_rng_state,
            restore_optimizer=not model_only,
            restore_scheduler=not model_only,
            restore_scaler=not model_only,
            strict_model=not model_only,
        )
        if not model_only:
            self.state = GPUTrainingState.from_dict(payload.get("trainer_state", {}))
            self.state.runtime_metadata = dict(self.runtime_meta)
        metadata.update(
            {
                "checkpoint_path": checkpoint_path,
                "resume_model_only": model_only,
            }
        )
        self.checkpoint_metadata = metadata
        self.model_metadata["resume"] = dict(metadata)
        return payload

    def _build_model(self):
        if self.config.model_ref:
            manifest = ModelRegistry().resolve_model_ref(self.config.model_ref)
            arch = manifest.architecture
        else:
            arch = ModelArchitectureConfig(
                name=self.config.name,
                model_type=self.config.model_type,
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                max_seq_len=self.config.max_seq_len,
                module_names=self.config.module_names or ["core", "coding", "debugging", "repair"],
                always_include_core=self.config.always_include_core,
                mop_block_type=self.config.mop_block_type,
                expert_count=self.config.expert_count,
                active_experts=self.config.active_experts,
                routing_granularity=self.config.routing_granularity,
                shared_depth_ratio=self.config.shared_depth_ratio,
                use_lora_deltas=self.config.use_lora_deltas,
                lora_rank=self.config.lora_rank,
                lora_target_modules=self.config.lora_target_modules,
                use_fast_adapters=self.config.use_fast_adapters,
                fast_adapter_names=self.config.fast_adapter_names,
                fast_adapter_bottleneck_dim=self.config.fast_adapter_bottleneck_dim,
                use_generated_params=self.config.use_generated_params,
                generated_condition_names=self.config.generated_condition_names,
                generated_condition_dim=self.config.generated_condition_dim,
                generated_rank=self.config.generated_rank,
                generated_type=self.config.generated_type,
                intended_scale="small_gpu",
            )
        model = build_tiny_model_from_architecture(arch, tokenizer=self.tokenizer)
        self.model_metadata["architecture"] = arch.to_dict()
        return model

    def _forward_loss(self, batch):
        target_modules = batch.get("target_modules")
        active_adapters = None
        active_conditions = None
        if self.config.use_fast_adapters:
            active_adapters = [adapter_names_from_target_modules(item) for item in (target_modules or [])]
        if self.config.use_generated_params:
            active_conditions = [condition_names_from_target_modules(item) for item in (target_modules or [])]
        with autocast_context(self.runtime):
            if "hidden_states" in batch:
                outputs = self.model.forward_from_hidden(
                    batch["hidden_states"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                    active_modules=target_modules,
                    active_adapters=active_adapters,
                    active_conditions=active_conditions,
                )
            else:
                kwargs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch.get("attention_mask"),
                    "labels": batch.get("labels"),
                }
                if self.config.model_type in {"mop_oracle", "mop_learned_router", "baseline_moe"}:
                    kwargs["active_modules"] = target_modules
                if active_adapters is not None:
                    kwargs["active_adapters"] = active_adapters
                if active_conditions is not None:
                    kwargs["active_conditions"] = active_conditions
                outputs = self.model(**kwargs)
        loss = outputs.get("loss")
        if loss is None:
            loss = outputs["logits"].sum() * 0.0
        fast_meta = fast_parameter_metadata(batch, self.model)
        active_meta = estimate_active_parameters(self.model, target_modules)
        self.model_metadata.update(
            {
                "active_module_density": active_meta.get("routing_density"),
                "active_adapter_density": fast_meta.get("active_adapter_density"),
                "generated_condition_density": fast_meta.get("generated_condition_density"),
                "active_param_estimate": active_meta.get("active_params"),
                "active_trainable_param_estimate": active_meta.get("active_trainable_params"),
                "shared_frozen_params": active_meta.get("shared_frozen_params"),
                "routed_module_params": active_meta.get("routed_module_params"),
                "active_expert_count": active_meta.get("active_expert_count"),
                "expert_count": active_meta.get("expert_count"),
                "expert_compute_ratio": active_meta.get("expert_compute_ratio"),
                "shared_compute_ratio": active_meta.get("shared_compute_ratio"),
                "estimated_active_flop_ratio": active_meta.get("estimated_active_flop_ratio"),
                "estimated_backward_flop_ratio": active_meta.get("estimated_backward_flop_ratio"),
                "frozen_prefix": dict(getattr(self.model, "last_forward_metadata", {}) or {}),
                "routing_mode": "oracle" if self.config.model_type != "dense" else "none",
            }
        )
        return loss

    def _append_metrics(self, extra: dict[str, Any]) -> None:
        snapshot = {
            "global_step": self.state.global_step,
            "optimizer_step": self.state.optimizer_step,
            "train_loss": self.state.latest_train_loss,
            "eval_loss": self.state.latest_eval_loss,
            "samples_seen": self.state.samples_seen,
            "tokens_seen": self.state.tokens_seen,
            "micro_batch_size": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": self.config.effective_batch_size,
            **extra,
        }
        self.state.metric_history.append(snapshot)

    def _memory_snapshot(self) -> dict[str, Any]:
        snapshot = {
            "step": self.state.global_step,
            "cuda_available": self.runtime_meta.get("cuda_available"),
            "selected_device": self.runtime_meta.get("selected_device"),
            "allocated_gb": None,
            "reserved_gb": None,
            "peak_allocated_gb": None,
            "peak_reserved_gb": None,
        }
        memory = cuda_memory_metrics(self.runtime)
        snapshot["allocated_gb"] = memory["final_allocated_gb"]
        snapshot["reserved_gb"] = memory["final_reserved_gb"]
        snapshot["peak_allocated_gb"] = memory["peak_allocated_gb"]
        snapshot["peak_reserved_gb"] = memory["peak_reserved_gb"]
        self.state.memory_snapshots.append(snapshot)
        return snapshot

    def _empty_cache(self) -> None:
        try:
            import torch

            if self.runtime.device_info.device_type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _write_setup_artifacts(self) -> None:
        _write_json(self.output_dir / "config.json", self.config.to_dict())
        _write_json(self.output_dir / "runtime.json", self.runtime_meta)
        estimate = estimate_from_config(self.config)
        memory_path = write_memory_estimate(estimate, self.output_dir / "memory_estimate.json")
        self.artifacts.update(
            {
                "config_json": str(self.output_dir / "config.json"),
                "runtime_json": str(self.output_dir / "runtime.json"),
                "memory_estimate_json": memory_path,
            }
        )

    def _write_state(self) -> None:
        _write_json(self.output_dir / "state.json", self.state.to_dict())

    def _finish(self, status: str) -> GPUTrainingResult:
        finite = self.state.latest_train_loss is None or math.isfinite(float(self.state.latest_train_loss))
        efficiency = self._efficiency_metrics()
        metrics = {
            "status": status,
            "finite": finite,
            "global_steps": self.state.global_step,
            "optimizer_steps": self.state.optimizer_step,
            "samples_seen": self.state.samples_seen,
            "tokens_seen": self.state.tokens_seen,
            "latest_train_loss": self.state.latest_train_loss,
            "latest_eval_loss": self.state.latest_eval_loss,
            "best_eval_loss": self.state.best_eval_loss,
            "evals_without_improvement": self.state.evals_without_improvement,
            "stopped_early": self.state.stopped_early,
            "stop_reason": self.state.stop_reason,
            "micro_batch_size": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": self.config.effective_batch_size,
            "runtime": dict(self.runtime_meta),
            "scaler": self.scaler.state_dict() if self.scaler is not None else {},
            "data": dict(self.data_metadata),
            "model": dict(self.model_metadata),
            "memory_snapshots": list(self.state.memory_snapshots),
            "efficiency": efficiency,
        }
        if self.config.run_generation_eval:
            metrics.update(self._generation_eval_metrics())
        _write_json(self.output_dir / "metrics.json", metrics)
        self.artifacts["metrics_json"] = str(self.output_dir / "metrics.json")
        if self.state.latest_checkpoint_path:
            self.artifacts["latest_checkpoint_path"] = self.state.latest_checkpoint_path
        result = GPUTrainingResult(
            run_id=self.run_id,
            status=status,
            config=self.config.to_dict(),
            state=self.state.to_dict(),
            metrics=metrics,
            artifacts=dict(self.artifacts),
            runtime_metadata=dict(self.runtime_meta),
            output_dir=str(self.output_dir),
        )
        result_path = result.save(self.output_dir / "gpu_training_result.json")
        self.artifacts["gpu_training_result_json"] = str(result_path)
        result.artifacts.update(self.artifacts)
        _write_json(result_path, result.to_dict())
        now = _now()
        self.registry.save_record(
            GPURunRecord(
                run_id=self.run_id,
                name=self.config.name,
                status=status,
                output_dir=str(self.output_dir),
                created_at=now,
                updated_at=now,
                latest_checkpoint_path=self.state.latest_checkpoint_path,
                metrics_path=str(self.output_dir / "metrics.json"),
                result_path=str(result_path),
                runtime_path=str(self.output_dir / "runtime.json"),
                metadata={
                    "model_type": self.config.model_type,
                    "selected_device": self.runtime_meta.get("selected_device"),
                    "selected_precision": self.runtime_meta.get("selected_precision"),
                },
            )
        )
        return result

    def _register_artifact(self, record: ArtifactRecord) -> None:
        try:
            self.artifact_manager.register(record)
        except ValueError:
            pass

    def _generation_eval_metrics(self) -> dict[str, Any]:
        if self.config.activation_cache_path:
            return {
                "generation_eval": {
                    "enabled": False,
                    "reason": "activation_cache_training_has_no_source_lessons",
                }
            }
        if self._data_config is None:
            return {
                "generation_eval": {
                    "enabled": False,
                    "reason": "data_config_missing",
                }
            }
        try:
            _, eval_lessons, _ = load_gpu_lesson_splits(self._data_config)
        except Exception as exc:
            return {
                "generation_eval": {
                    "enabled": False,
                    "error": str(exc),
                }
            }
        previous_mode = self.model.training
        self.model.eval()
        results = []
        try:
            for lesson in eval_lessons[: self.config.generation_eval_examples]:
                target_modules = list(lesson.target_modules)
                active_modules = (
                    target_modules
                    if self.config.model_type in {"mop_oracle", "mop_learned_router", "baseline_moe"}
                    else None
                )
                active_adapters = (
                    adapter_names_from_target_modules(target_modules)
                    if self.config.use_fast_adapters
                    else None
                )
                active_conditions = (
                    condition_names_from_target_modules(target_modules)
                    if self.config.use_generated_params
                    else None
                )
                results.append(
                    evaluate_generated_code_for_lesson(
                        self.model,
                        self.tokenizer,
                        lesson,
                        max_new_tokens=self.config.generation_max_new_tokens,
                        device=str(self.runtime.device_info.selected),
                        active_modules=active_modules,
                        active_adapters=active_adapters,
                        active_conditions=active_conditions,
                    )
                )
        finally:
            if previous_mode:
                self.model.train()
        summary = summarize_generation_results(results)
        return {
            "generation_eval": {
                "enabled": True,
                "examples": len(results),
                **summary,
            }
        }

    def _efficiency_metrics(self) -> dict[str, Any]:
        params = dict(self.model_metadata.get("parameter_counts") or {})
        total_params = int(params.get("total", 0) or 0)
        trainable_params = int(params.get("trainable", 0) or 0)
        frozen_params = int(params.get("frozen", max(0, total_params - trainable_params)) or 0)
        active_param_estimate = self.model_metadata.get("active_param_estimate")
        active_param_ratio = (
            float(active_param_estimate) / float(total_params)
            if active_param_estimate is not None and total_params > 0
            else None
        )
        active_trainable_param_estimate = self.model_metadata.get("active_trainable_param_estimate")
        active_trainable_param_ratio = (
            float(active_trainable_param_estimate) / float(total_params)
            if active_trainable_param_estimate is not None and total_params > 0
            else None
        )
        total_time = None
        if self._train_started_at is not None and self._train_finished_at is not None:
            total_time = max(0.0, self._train_finished_at - self._train_started_at)
        step_time = self._step_times[-1] if self._step_times else None
        samples_per_sec = (
            float(self.state.samples_seen) / total_time
            if total_time and total_time > 0
            else None
        )
        tokens_per_sec = (
            float(self.state.tokens_seen) / total_time
            if total_time and total_time > 0
            else None
        )
        cached_hidden_steps_per_sec = (
            float(self.state.global_step) / total_time
            if self.data_metadata.get("kind") == "activation_cache" and total_time and total_time > 0
            else None
        )
        memory = cuda_memory_metrics(self.runtime)
        checkpoint_size_mb = _file_size_mb(self.state.latest_checkpoint_path)
        return {
            "tokens_per_sec": _round_or_none(tokens_per_sec),
            "original_token_equivalent_tokens_per_sec": _round_or_none(tokens_per_sec),
            "cached_hidden_steps_per_sec": _round_or_none(cached_hidden_steps_per_sec),
            "activation_cache_enabled": self.data_metadata.get("kind") == "activation_cache",
            "samples_per_sec": _round_or_none(samples_per_sec),
            "step_time_sec": _round_or_none(step_time),
            "total_train_time_sec": _round_or_none(total_time),
            "peak_allocated_gb": memory["peak_allocated_gb"],
            "peak_reserved_gb": memory["peak_reserved_gb"],
            "final_allocated_gb": memory["final_allocated_gb"],
            "final_reserved_gb": memory["final_reserved_gb"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_param_ratio": (
                trainable_params / max(1, total_params)
                if total_params
                else None
            ),
            "active_param_estimate": active_param_estimate,
            "active_param_ratio": _round_or_none(active_param_ratio),
            "active_trainable_param_estimate": active_trainable_param_estimate,
            "active_trainable_param_ratio": _round_or_none(active_trainable_param_ratio),
            "shared_frozen_params": self.model_metadata.get("shared_frozen_params"),
            "routed_module_params": self.model_metadata.get("routed_module_params"),
            "active_expert_count": self.model_metadata.get("active_expert_count"),
            "expert_count": self.model_metadata.get("expert_count"),
            "expert_compute_ratio": self.model_metadata.get("expert_compute_ratio"),
            "shared_compute_ratio": self.model_metadata.get("shared_compute_ratio"),
            "estimated_active_flop_ratio": self.model_metadata.get("estimated_active_flop_ratio"),
            "estimated_backward_flop_ratio": self.model_metadata.get("estimated_backward_flop_ratio"),
            "frozen_prefix": dict(self.model_metadata.get("frozen_prefix") or {}),
            "active_module_density": self.model_metadata.get("active_module_density"),
            "active_adapter_density": self.model_metadata.get("active_adapter_density"),
            "generated_condition_density": self.model_metadata.get("generated_condition_density"),
            "checkpoint_size_mb": checkpoint_size_mb,
        }


def apply_activation_checkpointing(model, enabled: bool, policy: str = "auto") -> dict[str, Any]:
    """Enable model-native non-reentrant activation checkpointing."""

    if not enabled:
        if hasattr(model, "activation_checkpointing_enabled"):
            model.activation_checkpointing_enabled = False
        return {"enabled": False, "applied_block_count": 0, "policy": policy, "warnings": []}
    blocks = 0
    for name in ("blocks", "shared_blocks", "routed_blocks"):
        module = getattr(model, name, None)
        if module is not None:
            try:
                blocks += len(module.layers)
            except Exception:
                try:
                    blocks += len(module)
                except Exception:
                    pass
    supported = hasattr(model, "activation_checkpointing_enabled")
    if supported:
        model.activation_checkpointing_enabled = True
    return {
        "enabled": True,
        "applied_block_count": blocks if supported else 0,
        "candidate_block_count": blocks,
        "policy": policy,
        "warnings": (
            []
            if supported
            else ["Model does not expose native activation checkpointing support."]
        ),
    }


def select_attention_metadata(efficient_attention: str) -> dict[str, Any]:
    try:
        import torch

        has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    except Exception:
        has_sdpa = False
    selected = "torch_sdpa" if efficient_attention in {"auto", "torch_sdpa"} and has_sdpa else "eager"
    warning = []
    if efficient_attention == "torch_sdpa" and not has_sdpa:
        warning.append("torch_sdpa requested but unavailable; using eager attention.")
    return {"requested": efficient_attention, "selected": selected, "warnings": warning}


def _runtime_config(config: GPUTrainingConfig) -> RuntimeConfig:
    return RuntimeConfig(
        device=config.device,
        precision=config.precision,
        enable_amp=config.enable_amp,
        allow_tf32=config.allow_tf32,
        deterministic=config.deterministic,
        compile_model=config.compile_model,
        require_device_available=config.require_device_available,
    )


def _build_scheduler(optimizer, config: GPUTrainingConfig):
    if config.scheduler == "none":
        return None
    try:
        import torch
    except Exception:
        return None

    def lr_lambda(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return min(1.0, float(step + 1) / float(max(1, config.warmup_steps)))
        if config.scheduler == "cosine":
            decay_steps = max(1, config.max_steps - config.warmup_steps)
            progress = min(
                1.0,
                float(max(0, step - config.warmup_steps)) / float(decay_steps),
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if config.scheduler == "linear_warmup":
            return 1.0
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _max_examples(config: GPUTrainingConfig) -> int | None:
    values = [value for value in (config.max_train_examples, config.max_eval_examples) if value is not None]
    return max(values) if values else None


def _batch_size(batch: dict[str, Any]) -> int:
    value = batch.get("input_ids")
    if value is None:
        value = batch.get("hidden_states")
    return int(value.shape[0]) if hasattr(value, "shape") else 0


def _token_count(batch: dict[str, Any]) -> int:
    mask = batch.get("attention_mask")
    if hasattr(mask, "sum"):
        return int(mask.sum().detach().cpu().item())
    cached = batch.get("cached_token_count")
    if isinstance(cached, int):
        return cached
    value = batch.get("input_ids")
    return int(value.numel()) if hasattr(value, "numel") else 0


def _resolve_checkpoint_reference(ref: str, output_root: str) -> str:
    candidate = Path(ref)
    if candidate.exists():
        return str(candidate)
    return GPURunRegistry(output_root).latest_checkpoint(ref)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


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


def _make_run_id(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    slug = "-".join(part for part in slug.split("-") if part) or "gpu-train"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slug}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GPUTrainer.") from exc
    return torch
