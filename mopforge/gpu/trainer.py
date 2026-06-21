"""Single-device GPU-aware trainer with CPU fallback."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
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
from mopforge.eval import (
    evaluate_ground_truth_controls,
    evaluate_generated_code_for_lesson,
    select_generation_eval_lessons,
    summarize_generation_results,
)
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
        self._last_loss_metadata: dict[str, Any] = {}
        self._latest_eval_metrics: dict[str, Any] = {}
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
            shuffle_train=self.config.shuffle_train,
            shuffle_seed=self.config.train_shuffle_seed,
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
                shuffle_train=self.config.shuffle_train,
                shuffle_seed=self.config.train_shuffle_seed,
                hard_example_replay_enabled=self.config.hard_example_replay_enabled,
                hard_example_replay_loss_threshold=self.config.hard_example_replay_loss_threshold,
                hard_example_replay_multiplier=self.config.hard_example_replay_multiplier,
            )
        else:
            self.train_loader, self.eval_loader, self.data_metadata = build_gpu_dataloaders(
                data_config,
                self.tokenizer,
                self.runtime,
            )
        self._train_iter = iter(self.train_loader)

        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        if self.config.activation_cache_path and self.config.offload_frozen_backbone_for_cache:
            self.model_metadata["cached_training_backbone_offload"] = offload_cached_frozen_backbone(
                self.model,
                runtime=self.runtime,
            )

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
                batch = move_batch_to_device(
                    self._next_train_batch(),
                    self.runtime.device_info.selected,
                )
                loss = self._forward_loss(batch, include_distillation=True)
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
                        self._maybe_record_target_eval_loss(self.state.latest_eval_loss)
                        if self.config.save_full_checkpoints and self.config.save_best_eval_checkpoint:
                            self.save_checkpoint(
                                self.state.global_step,
                                tag="best-eval",
                                record_latest=False,
                                record_best=True,
                        )
                    else:
                        self.state.evals_without_improvement += 1
                        self._maybe_record_target_eval_loss(self.state.latest_eval_loss)
                    if (
                        self.config.early_stopping_enabled
                        and self.state.evals_without_improvement
                        >= self.config.early_stopping_patience_evals
                    ):
                        self.state.stopped_early = True
                        self.state.stop_reason = "eval_loss_patience_exhausted"
                        break
                if self.state.global_step % self.config.log_every_steps == 0:
                    self._append_metrics({"event": "log", **dict(self._last_loss_metadata)})
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
            self._empty_cache()
            self._write_state()
        return self._finish(status)

    def evaluate(self) -> dict[str, Any]:
        if not self._setup_done:
            self.setup()
        losses: list[float] = []
        examples = 0
        self.model.eval()
        try:
            import torch

            with torch.no_grad():
                for index, batch in enumerate(self.eval_loader):
                    if not self.config.eval_full_dataset and index >= self.config.eval_batches:
                        break
                    batch = move_batch_to_device(batch, self.runtime.device_info.selected)
                    loss = self._forward_loss(batch, include_distillation=False)
                    losses.append(float(loss.detach().float().cpu().item()))
                    examples += _batch_size(batch)
        finally:
            self.model.train()
        mean_loss = sum(losses) / len(losses) if losses else None
        metrics = {
            "eval_loss_mean": mean_loss,
            "eval_batches": len(losses),
            "eval_examples": examples,
            "eval_full_dataset": bool(self.config.eval_full_dataset),
            "step": self.state.global_step,
        }
        self._latest_eval_metrics = dict(metrics)
        self._append_metrics(metrics)
        return metrics

    def _next_train_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        if self.state.train_epoch <= 0:
            self.state.train_epoch = 1
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self.state.train_epoch += 1
            self.state.train_batches_in_epoch = 0
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        self.state.train_batches_in_epoch += 1
        return batch

    def save_checkpoint(
        self,
        step: int,
        *,
        tag: str | None = None,
        record_latest: bool = True,
        record_best: bool = False,
    ) -> str:
        filename = f"checkpoint-{tag}.pt" if tag else f"checkpoint-step-{step:06d}.pt"
        path = self.checkpoint_dir / filename
        if record_best:
            self.state.best_checkpoint_path = str(path)
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
        if record_latest:
            self.state.latest_checkpoint_path = saved
        if record_best:
            self.state.best_checkpoint_path = saved
        self.checkpoint_metadata = {
            "latest_checkpoint_path": self.state.latest_checkpoint_path,
            "best_checkpoint_path": self.state.best_checkpoint_path,
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
                    "checkpoint_tag": tag,
                    "record_latest": record_latest,
                    "record_best": record_best,
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
                lora_tail_only=self.config.lora_tail_only,
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

    def _forward_loss(self, batch, *, include_distillation: bool = True):
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
        ce_loss = loss
        distillation_loss = None
        if include_distillation:
            distillation_loss = self._distillation_loss(batch, outputs)
            if distillation_loss is not None:
                loss = loss + float(self.config.distillation_weight) * distillation_loss
        self._last_loss_metadata = {
            "ce_loss": _float_or_none(ce_loss),
            "distillation_loss": _float_or_none(distillation_loss),
            "distillation_enabled": bool(
                self.config.distillation_enabled
                and self.config.distillation_weight > 0
                and distillation_loss is not None
            ),
        }
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
                "distillation": self._distillation_metadata(batch),
                "hard_example_replay": self._hard_example_replay_metadata(),
            }
        )
        return loss

    def _distillation_loss(self, batch, outputs):
        if not self.config.distillation_enabled or self.config.distillation_weight <= 0:
            return None
        if "teacher_topk_logits" not in batch or "teacher_topk_indices" not in batch:
            return None
        torch = _require_torch()
        teacher_logits = batch["teacher_topk_logits"].to(outputs["logits"].device).float()
        teacher_indices = batch["teacher_topk_indices"].to(outputs["logits"].device).long()
        student_logits = outputs["logits"].float().gather(-1, teacher_indices)
        temperature = float(self.config.distillation_temperature)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        token_kl = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1) * (temperature**2)
        mask = batch.get("labels")
        if mask is not None:
            mask = (mask != -100).to(device=token_kl.device, dtype=token_kl.dtype)
        else:
            attention_mask = batch.get("attention_mask")
            mask = (
                attention_mask.to(device=token_kl.device, dtype=token_kl.dtype)
                if attention_mask is not None
                else torch.ones_like(token_kl)
            )
        denominator = mask.sum().clamp_min(1.0)
        return (token_kl * mask).sum() / denominator

    def _distillation_metadata(self, batch) -> dict[str, Any]:
        cache_metadata = dict(self.data_metadata.get("cache_metadata") or {})
        return {
            "enabled": bool(self.config.distillation_enabled),
            "weight": float(self.config.distillation_weight),
            "temperature": float(self.config.distillation_temperature),
            "configured_top_k": int(self.config.distillation_top_k),
            "batch_has_teacher_topk": bool(
                "teacher_topk_logits" in batch and "teacher_topk_indices" in batch
            ),
            "cache_teacher_top_k": cache_metadata.get("teacher_top_k"),
            "cache_distillation_ready": bool(cache_metadata.get("distillation_ready")),
        }

    def _hard_example_replay_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.hard_example_replay_enabled),
            "loss_threshold": self.config.hard_example_replay_loss_threshold,
            "multiplier": int(self.config.hard_example_replay_multiplier),
            **dict(self.data_metadata.get("hard_example_replay") or {}),
        }

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

    def _maybe_record_target_eval_loss(self, eval_loss: float | None) -> None:
        target = self.config.target_eval_loss
        if target is None or self.state.target_eval_loss_reached or eval_loss is None:
            return
        if float(eval_loss) > float(target):
            return
        elapsed = None
        if self._train_started_at is not None:
            elapsed = max(0.0, time.perf_counter() - self._train_started_at)
        memory = self._memory_snapshot()
        self.state.target_eval_loss_reached = True
        self.state.target_eval_loss_value = float(eval_loss)
        self.state.target_eval_loss_step = int(self.state.global_step)
        self.state.target_eval_loss_samples_seen = int(self.state.samples_seen)
        self.state.target_eval_loss_tokens_seen = int(self.state.tokens_seen)
        self.state.target_eval_loss_time_sec = _round_or_none(elapsed)
        self.state.target_eval_loss_memory_snapshot = dict(memory)
        self._append_metrics(
            {
                "event": "target_eval_loss_reached",
                "target_eval_loss": float(target),
                "target_eval_loss_value": float(eval_loss),
                "time_to_target_loss_sec": self.state.target_eval_loss_time_sec,
                "tokens_to_target_loss": self.state.target_eval_loss_tokens_seen,
                "samples_to_target_loss": self.state.target_eval_loss_samples_seen,
                "target_peak_allocated_gb": memory.get("peak_allocated_gb"),
                "target_peak_reserved_gb": memory.get("peak_reserved_gb"),
            }
        )

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
            "train_epoch": self.state.train_epoch,
            "train_batches_in_epoch": self.state.train_batches_in_epoch,
            "samples_seen": self.state.samples_seen,
            "tokens_seen": self.state.tokens_seen,
            "latest_train_loss": self.state.latest_train_loss,
            "latest_eval_loss": self.state.latest_eval_loss,
            "best_eval_loss": self.state.best_eval_loss,
            "latest_eval_batches": self._latest_eval_metrics.get("eval_batches"),
            "latest_eval_examples": self._latest_eval_metrics.get("eval_examples"),
            "eval_full_dataset": self.config.eval_full_dataset,
            "evals_without_improvement": self.state.evals_without_improvement,
            "target_eval_loss": self.config.target_eval_loss,
            "target_eval_loss_reached": self.state.target_eval_loss_reached,
            "target_eval_loss_value": self.state.target_eval_loss_value,
            "target_eval_loss_step": self.state.target_eval_loss_step,
            "target_eval_loss_samples_seen": self.state.target_eval_loss_samples_seen,
            "target_eval_loss_tokens_seen": self.state.target_eval_loss_tokens_seen,
            "target_eval_loss_time_sec": self.state.target_eval_loss_time_sec,
            "stopped_early": self.state.stopped_early,
            "stop_reason": self.state.stop_reason,
            "micro_batch_size": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": self.config.effective_batch_size,
            "shuffle_train": self.config.shuffle_train,
            "train_shuffle_seed": self.config.train_shuffle_seed,
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
        if self.state.best_checkpoint_path:
            self.artifacts["best_checkpoint_path"] = self.state.best_checkpoint_path
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
                    "best_checkpoint_path": self.state.best_checkpoint_path,
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
        if self._data_config is None:
            return {
                "generation_eval": {
                    "enabled": False,
                    "reason": "data_config_missing",
                }
            }
        try:
            train_lessons, eval_lessons, _ = load_gpu_lesson_splits(self._data_config)
        except Exception as exc:
            return {
                "generation_eval": {
                    "enabled": False,
                    "error": str(exc),
                }
            }
        cached_training = bool(self.config.activation_cache_path)
        if cached_training:
            self.model = move_model_to_runtime(self.model, self.runtime)
        try:
            checkpoint = self._restore_generation_eval_checkpoint()
        except Exception as exc:
            if cached_training and self.config.offload_frozen_backbone_for_cache:
                offload_cached_frozen_backbone(self.model, runtime=self.runtime)
                self._empty_cache()
            return {
                "generation_eval": {
                    "enabled": False,
                    "error": f"best-checkpoint restore failed: {exc}",
                    "checkpoint_source": "restore_failed",
                }
            }
        previous_mode = self.model.training
        self.model.eval()
        selected_eval = select_generation_eval_lessons(
            eval_lessons,
            max_lessons=self.config.generation_eval_examples,
            stratify_by=self.config.generation_eval_stratify_by,
        )
        selected_by_split = {"eval": selected_eval}
        if self.config.generation_eval_include_train:
            selected_by_split["train"] = select_generation_eval_lessons(
                train_lessons,
                max_lessons=self.config.generation_eval_examples,
                stratify_by=self.config.generation_eval_stratify_by,
            )
        results_by_split: dict[str, list[dict[str, Any]]] = {}
        try:
            for split_name, lessons in selected_by_split.items():
                results_by_split[split_name] = self._generate_for_lessons(lessons)
        finally:
            if previous_mode:
                self.model.train()
            if cached_training and self.config.offload_frozen_backbone_for_cache:
                offload_cached_frozen_backbone(self.model, runtime=self.runtime)
                self._empty_cache()
        summaries = {
            split_name: summarize_generation_results(results)
            for split_name, results in results_by_split.items()
        }
        results = results_by_split["eval"]
        summary = summaries["eval"]
        ground_truth_controls = evaluate_ground_truth_controls(eval_lessons)
        controls_path = self.output_dir / "ground_truth_controls.json"
        _write_json(controls_path, ground_truth_controls)
        generation_path = self.output_dir / "generation_eval.json"
        measurement_scope = (
            "post_training_full_model_quality_eval"
            if cached_training
            else "post_training_quality_eval"
        )
        _write_json(
            generation_path,
            {
                "measurement_scope": measurement_scope,
                "checkpoint": checkpoint,
                "selection": {
                    "stratify_by": self.config.generation_eval_stratify_by,
                    "requested_examples_per_split": self.config.generation_eval_examples,
                    "max_new_tokens": self.config.generation_max_new_tokens,
                    "selected_categories": {
                        split_name: dict(
                            sorted(
                                Counter(
                                    str(lesson.metadata.get("bug_type") or lesson.subskill or "unknown")
                                    for lesson in lessons
                                ).items()
                            )
                        )
                        for split_name, lessons in selected_by_split.items()
                    },
                },
                "summary": summary,
                "results": results,
                "splits": {
                    split_name: {
                        "summary": summaries[split_name],
                        "results": split_results,
                    }
                    for split_name, split_results in results_by_split.items()
                },
                "ground_truth_controls": {
                    "passed": ground_truth_controls["passed"],
                    "examples": ground_truth_controls["examples"],
                    "artifact_path": str(controls_path),
                },
            },
        )
        self.artifacts["generation_eval_json"] = str(generation_path)
        self.artifacts["ground_truth_controls_json"] = str(controls_path)
        return {
            "generation_eval": {
                "enabled": True,
                "examples": len(results),
                "measurement_scope": measurement_scope,
                "artifact_path": str(generation_path),
                "checkpoint_source": checkpoint["source"],
                "checkpoint_path": checkpoint.get("path"),
                "checkpoint_global_step": checkpoint.get("global_step"),
                "stratify_by": self.config.generation_eval_stratify_by,
                "split_summaries": summaries,
                "ground_truth_controls_passed": ground_truth_controls["passed"],
                "ground_truth_controls_examples": ground_truth_controls["examples"],
                "ground_truth_controls_path": str(controls_path),
                **summary,
            }
        }

    def _restore_generation_eval_checkpoint(self) -> dict[str, Any]:
        """Restore the best eval weights without changing final trainer state."""

        path = self.state.best_checkpoint_path
        if not self.config.generation_eval_use_best_checkpoint or not path:
            return {
                "source": "final",
                "path": self.state.latest_checkpoint_path,
                "global_step": self.state.global_step,
                "reason": (
                    "best_checkpoint_disabled"
                    if not self.config.generation_eval_use_best_checkpoint
                    else "best_checkpoint_unavailable"
                ),
            }
        payload = load_gpu_checkpoint(path, map_location="cpu")
        restored = restore_gpu_checkpoint(
            payload,
            model=self.model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            restore_rng=False,
            restore_optimizer=False,
            restore_scheduler=False,
            restore_scaler=False,
            strict_model=False,
        )
        checkpoint_state = dict(payload.get("trainer_state") or {})
        return {
            "source": "best_eval",
            "path": str(path),
            "global_step": checkpoint_state.get("global_step"),
            "optimizer_step": checkpoint_state.get("optimizer_step"),
            "best_eval_loss": checkpoint_state.get("best_eval_loss"),
            "restore": restored,
        }

    def _generate_for_lessons(self, lessons) -> list[dict[str, Any]]:
        results = []
        for lesson in lessons:
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
        return results

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
        target_memory = dict(self.state.target_eval_loss_memory_snapshot or {})
        checkpoint_size_mb = _file_size_mb(self.state.latest_checkpoint_path)
        return {
            "tokens_per_sec": _round_or_none(tokens_per_sec),
            "original_token_equivalent_tokens_per_sec": _round_or_none(tokens_per_sec),
            "cached_hidden_steps_per_sec": _round_or_none(cached_hidden_steps_per_sec),
            "activation_cache_enabled": self.data_metadata.get("kind") == "activation_cache",
            "samples_per_sec": _round_or_none(samples_per_sec),
            "step_time_sec": _round_or_none(step_time),
            "total_train_time_sec": _round_or_none(total_time),
            "target_eval_loss": self.config.target_eval_loss,
            "target_eval_loss_reached": self.state.target_eval_loss_reached,
            "target_eval_loss_value": self.state.target_eval_loss_value,
            "time_to_target_loss_sec": self.state.target_eval_loss_time_sec,
            "tokens_to_target_loss": self.state.target_eval_loss_tokens_seen,
            "samples_to_target_loss": self.state.target_eval_loss_samples_seen,
            "target_peak_allocated_gb": target_memory.get("peak_allocated_gb"),
            "target_peak_reserved_gb": target_memory.get("peak_reserved_gb"),
            "target_final_allocated_gb": target_memory.get("allocated_gb"),
            "target_final_reserved_gb": target_memory.get("reserved_gb"),
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
            "distillation_enabled": self.model_metadata.get("distillation", {}).get("enabled"),
            "distillation_weight": self.model_metadata.get("distillation", {}).get("weight"),
            "distillation_temperature": self.model_metadata.get("distillation", {}).get("temperature"),
            "distillation_top_k": self.model_metadata.get("distillation", {}).get("cache_teacher_top_k"),
            "hard_example_replay_enabled": self.model_metadata.get("hard_example_replay", {}).get("enabled"),
            "hard_example_replay_multiplier": self.model_metadata.get("hard_example_replay", {}).get("multiplier"),
            "hard_example_count": self.model_metadata.get("hard_example_replay", {}).get("hard_example_count"),
            "hard_replayed_example_count": self.model_metadata.get("hard_example_replay", {}).get("replayed_example_count"),
            "cached_backbone_offloaded_param_count": self.model_metadata.get(
                "cached_training_backbone_offload",
                {},
            ).get("offloaded_param_count"),
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


def offload_cached_frozen_backbone(model, *, runtime) -> dict[str, Any]:
    """Move unused frozen cached-training backbone modules off CUDA."""

    candidate_names = [
        "token_embedding",
        "position_embedding",
        "blocks",
        "shared_blocks",
        "routed_blocks",
        "module_bank",
    ]
    offloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for name in candidate_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        param_count = _module_param_count(module)
        if _module_has_trainable_parameters(module):
            skipped.append(
                {
                    "name": name,
                    "reason": "has_trainable_parameters",
                    "param_count": param_count,
                }
            )
            continue
        try:
            module.to("cpu")
            offloaded.append({"name": name, "param_count": param_count})
        except Exception as exc:
            skipped.append(
                {
                    "name": name,
                    "reason": f"offload_failed:{exc}",
                    "param_count": param_count,
                }
            )
    cuda_cleanup = False
    try:
        import torch

        if runtime.device_info.device_type == "cuda":
            torch.cuda.empty_cache()
            cuda_cleanup = True
    except Exception:
        cuda_cleanup = False
    return {
        "enabled": True,
        "selected_device": getattr(runtime.device_info, "selected", None),
        "offloaded_modules": offloaded,
        "skipped_modules": skipped,
        "offloaded_param_count": sum(int(item["param_count"]) for item in offloaded),
        "skipped_param_count": sum(int(item["param_count"]) for item in skipped),
        "cuda_empty_cache_after_offload": cuda_cleanup,
        "note": "Only modules unused by forward_from_hidden are offloaded.",
    }


def _module_has_trainable_parameters(module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters(recurse=True))


def _module_param_count(module) -> int:
    try:
        return sum(int(parameter.numel()) for parameter in module.parameters(recurse=True))
    except Exception:
        return 0


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


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().item()
        return float(value)
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
