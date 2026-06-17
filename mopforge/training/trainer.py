"""Reusable CPU-first tiny trainer skeleton."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.artifacts import ArtifactManager, ArtifactRecord, CheckpointManager
from mopforge.curriculum import CurriculumConfig, CurriculumScheduler
from mopforge.data import CausalLMCollator, LessonCausalLMDataset, RouterCollator, RouterDataset
from mopforge.eval import evaluate_generated_code_for_lesson, summarize_generation_results
from mopforge.experiments.utils import set_seed, split_lessons
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.lifecycle import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
    restore_rng_state,
)
from mopforge.models import (
    TinyCausalTransformer,
    TinyMoPCausalTransformer,
    TinyModuleRouter,
    adapter_names_from_target_modules,
    condition_names_from_target_modules,
)
from mopforge.queues import TrainingQueueStore
from mopforge.runs import RunRegistry, TrainingRunRecord
from mopforge.runtime import (
    RuntimeConfig,
    apply_runtime_determinism,
    autocast_context,
    build_runtime_context,
    move_batch_to_device,
    move_model_to_runtime,
    runtime_metadata,
)
from mopforge.tokenization import (
    TokenizerProtocol,
    TokenizerSpec,
    build_tokenizer,
    get_tokenizer_pad_token_id,
    get_tokenizer_vocab_size,
    tokenizer_spec_from_config,
)
from mopforge.training.parameter_policy import (
    ParameterGroupSummary,
    TrainableParameterPolicy,
    apply_trainable_policy,
    count_parameters,
)
from mopforge.training.routing import DEFAULT_KNOWN_MODULES, route_batch_with_router
from mopforge.training.state import TrainerConfig, TrainerResult, TrainerState


class TinyTrainer:
    """A small production-style trainer skeleton with CPU-safe defaults."""

    def __init__(self, config: TrainerConfig) -> None:
        """Create a trainer for ``config``."""

        self.config = config
        self.state = TrainerState()
        self.run_id = _make_run_id(config.run_name)
        self.registry = RunRegistry(config.run_registry_root)
        self.run_dir = self.registry.create_run_dir(self.run_id)
        self.artifact_manager = ArtifactManager(config.artifact_root)
        self.checkpoint_manager = CheckpointManager(self.artifact_manager)
        self.tokenizer_spec = tokenizer_spec_from_config(config)
        self.tokenizer = build_tokenizer(self.tokenizer_spec)
        self.tokenizer_spec = _spec_from_tokenizer(self.tokenizer, self.tokenizer_spec)
        self.device = None
        self.runtime = None
        self.indexed_store: IndexedLessonStore | None = None
        self.plan = None
        self.lessons: list[KnowledgeLesson] = []
        self.train_lessons: list[KnowledgeLesson] = []
        self.eval_lessons: list[KnowledgeLesson] = []
        self.model = None
        self.router = None
        self.optimizer = None
        self.scheduler = None
        self.trainable_policy: TrainableParameterPolicy | None = None
        self.parameter_group_summaries: list[dict[str, Any]] = []
        self.parameter_counts: dict[str, int] = {}
        self.adapter_metadata: dict[str, Any] = {}
        self.generated_metadata: dict[str, Any] = {}
        self.train_loader = None
        self.eval_loader = None
        self.router_train_loader = None
        self.router_eval_loader = None
        self._train_iter: Iterator | None = None
        self._router_train_iter: Iterator | None = None
        self._setup_done = False
        self.artifacts: dict[str, Any] = {}
        self.queue_metadata: dict[str, Any] = {}
        self.resume_metadata: dict[str, Any] = {}
        self._full_checkpoint_steps: set[int] = set()

    def setup(self) -> None:
        """Load data, build curriculum, create model/optimizer, and resume if needed."""

        torch = _require_torch()
        _require_collators()
        set_seed(self.config.seed)
        self.runtime = build_runtime_context(_runtime_config_from_trainer(self.config))
        apply_runtime_determinism(self.runtime, self.config.seed)
        self.device = torch.device(self.runtime.device_info.selected)
        self._write_tokenizer_spec_artifact()

        self.indexed_store = IndexedLessonStore(
            self.config.lesson_path,
            self.config.index_path,
            auto_rebuild=True,
        )
        scheduler = CurriculumScheduler(indexed_store=self.indexed_store)
        self.plan = scheduler.build_plan(
            CurriculumConfig(
                strategy=self.config.curriculum_strategy,
                batch_size=self.config.batch_size,
                domains=self.config.curriculum_domains,
                skills=self.config.curriculum_skills,
                target_modules=self.config.target_modules,
                verification_statuses=self.config.curriculum_verification_statuses,
                feedback_store_path=self.config.feedback_store_path,
            )
        )
        plan_path = self.plan.save_json(self.run_dir / "curriculum_plan.json")
        self.artifacts["curriculum_plan_json"] = str(plan_path)
        self._register_artifact(
            ArtifactRecord(
                artifact_id=f"trainer-curriculum-{self.run_id}",
                kind="curriculum_plan",
                path=str(plan_path),
                run_id=self.run_id,
                model_type=self.config.model_type,
                metadata={"strategy": self.config.curriculum_strategy},
            )
        )

        self.lessons = scheduler.load_lessons(self.plan)
        if not self.lessons:
            raise ValueError("Trainer curriculum produced no lessons.")
        self.train_lessons, self.eval_lessons = _split_for_trainer(self.lessons, self.config.seed)
        self.train_loader = _lm_loader(self.train_lessons, self.tokenizer, self.config)
        self.eval_loader = _lm_loader(self.eval_lessons, self.tokenizer, self.config)
        self._train_iter = cycle(self.train_loader)

        if self.config.model_type == "dense":
            self.model = TinyCausalTransformer(
                vocab_size=get_tokenizer_vocab_size(self.tokenizer),
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                max_seq_len=self.config.max_seq_len,
                use_fast_adapters=self.config.use_fast_adapters,
                fast_adapter_names=self.config.fast_adapter_names,
                fast_adapter_bottleneck_dim=self.config.fast_adapter_bottleneck_dim,
                use_generated_params=self.config.use_generated_params,
                generated_condition_names=self.config.generated_condition_names,
                generated_condition_dim=self.config.generated_condition_dim,
                generated_rank=self.config.generated_rank,
                generated_type=self.config.generated_type,
            )
        elif self.config.model_type in {"mop_oracle", "mop_learned_router"}:
            self.model = TinyMoPCausalTransformer(
                vocab_size=get_tokenizer_vocab_size(self.tokenizer),
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                max_seq_len=self.config.max_seq_len,
                module_names=DEFAULT_KNOWN_MODULES,
                use_fast_adapters=self.config.use_fast_adapters,
                fast_adapter_names=self.config.fast_adapter_names,
                fast_adapter_bottleneck_dim=self.config.fast_adapter_bottleneck_dim,
                use_generated_params=self.config.use_generated_params,
                generated_condition_names=self.config.generated_condition_names,
                generated_condition_dim=self.config.generated_condition_dim,
                generated_rank=self.config.generated_rank,
                generated_type=self.config.generated_type,
            )
            if self.config.model_type == "mop_learned_router":
                self.router = TinyModuleRouter(
                    vocab_size=get_tokenizer_vocab_size(self.tokenizer),
                    d_model=self.config.d_model,
                    hidden_dim=max(32, self.config.d_model * 2),
                    known_modules=DEFAULT_KNOWN_MODULES,
                    pad_token_id=get_tokenizer_pad_token_id(self.tokenizer),
                )
                self.router_train_loader = _router_loader(
                    self.train_lessons,
                    self.tokenizer,
                    self.config,
                )
                self.router_eval_loader = _router_loader(
                    self.eval_lessons,
                    self.tokenizer,
                    self.config,
                )
                self._router_train_iter = cycle(self.router_train_loader)
        else:
            raise ValueError(f"Unsupported model_type: {self.config.model_type}")

        if self.model is None:
            raise RuntimeError("Model was not created.")
        self.model = move_model_to_runtime(self.model, self.runtime)
        if self.router is not None:
            self.router = move_model_to_runtime(self.router, self.runtime)
        self.state.runtime_metadata = runtime_metadata(self.runtime)
        self._apply_parameter_policy()
        self.adapter_metadata = self._adapter_metadata()
        self.generated_metadata = self._generated_metadata()
        self.optimizer = torch.optim.AdamW(
            self._trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.queue_metadata = self._queue_metadata()
        if self.config.resume_from_checkpoint:
            self.load_full_checkpoint(self.config.resume_from_checkpoint)
        elif self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        self._setup_done = True

    def train(self) -> TrainerResult:
        """Run the tiny train/eval/checkpoint loop and return a result."""

        if not self._setup_done:
            self.setup()
        while self.state.global_step < self.config.max_steps:
            train_loss = self._train_step()
            self.state.global_step += 1
            self.state.latest_train_loss = train_loss
            metrics: dict[str, Any] = {
                "step": self.state.global_step,
                "train_loss": train_loss,
            }

            if self.state.global_step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)
                eval_loss = eval_metrics["eval_loss_mean"]
                self.state.latest_eval_loss = eval_loss
                if self.state.best_eval_loss is None or eval_loss < self.state.best_eval_loss:
                    self.state.best_eval_loss = eval_loss

            if (
                self.config.save_checkpoints
                and self.state.global_step % self.config.checkpoint_interval == 0
            ):
                checkpoint = self.save_checkpoint(self.state.global_step)
                if checkpoint is not None:
                    self.state.checkpoint_artifacts.append(checkpoint.artifact_id)
                    metrics["checkpoint_artifact_id"] = checkpoint.artifact_id

            if (
                self.config.save_full_checkpoints
                and self.config.checkpoint_every_steps is not None
                and self.state.global_step % self.config.checkpoint_every_steps == 0
            ):
                full_checkpoint = self.save_full_checkpoint(
                    self.state.global_step,
                    reason="interval",
                )
                if full_checkpoint is not None:
                    metrics["full_checkpoint_artifact_id"] = full_checkpoint.artifact_id

            self.state.metrics_history.append(metrics)

        if self.state.latest_eval_loss is None:
            eval_metrics = self.evaluate()
            self.state.latest_eval_loss = eval_metrics["eval_loss_mean"]
            self.state.metrics_history.append({"step": self.state.global_step, **eval_metrics})

        if (
            self.config.save_full_checkpoints
            and self.state.global_step not in self._full_checkpoint_steps
        ):
            self.save_full_checkpoint(self.state.global_step, reason="final")

        final_metrics = self._final_metrics()
        if self.config.run_generation_eval:
            final_metrics.update(self._generation_eval_metrics())
        run_record = TrainingRunRecord(
            run_id=self.run_id,
            run_name=self.config.run_name,
            model_type=self.config.model_type,
            curriculum_strategy=self.config.curriculum_strategy,
            started_at=_now(),
            finished_at=_now(),
            config=self.config.to_dict(),
            metrics=final_metrics,
            artifacts=dict(self.artifacts),
        )
        run_json_path = self.registry.save(run_record)
        self.artifacts.update(run_record.artifacts)
        self._register_artifact(
            ArtifactRecord(
                artifact_id=f"trainer-run-metrics-{self.run_id}",
                kind="metrics",
                path=str(self.run_dir / "metrics.json"),
                run_id=self.run_id,
                model_type=self.config.model_type,
                metadata={"source": "TinyTrainer"},
            )
        )

        state_path = self._write_json("trainer_state.json", self.state.to_dict())
        self.artifacts["trainer_state_json"] = str(state_path)
        self._register_artifact(
            ArtifactRecord(
                artifact_id=f"trainer-state-{self.run_id}",
                kind="config",
                path=str(state_path),
                run_id=self.run_id,
                model_type=self.config.model_type,
            )
        )

        result = TrainerResult(
            run_id=self.run_id,
            run_name=self.config.run_name,
            model_type=self.config.model_type,
            routing_mode=self.config.routing_mode,
            final_state=self.state.to_dict(),
            metrics=final_metrics,
            artifacts={**self.artifacts, "run_json": str(run_json_path)},
            finite=bool(final_metrics["finite"]),
        )
        result_path = result.save_json(self.run_dir / "trainer_result.json")
        result.artifacts["trainer_result_json"] = str(result_path)
        result.save_json(result_path)
        self.artifacts["trainer_result_json"] = str(result_path)
        self._register_artifact(
            ArtifactRecord(
                artifact_id=f"trainer-result-{self.run_id}",
                kind="config",
                path=str(result_path),
                run_id=self.run_id,
                model_type=self.config.model_type,
            )
        )
        return result

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the current model for a few CPU-safe batches."""

        if self.model is None or self.eval_loader is None:
            raise RuntimeError("Trainer must be set up before evaluate().")
        torch = _require_torch()
        self.model.eval()
        if self.router is not None:
            self.router.eval()
        losses: list[float] = []
        router_iter = iter(self.router_eval_loader) if self.router_eval_loader is not None else None
        with torch.no_grad():
            for batch_index, batch in enumerate(self.eval_loader, start=1):
                batch = _move_lm_batch(batch, self.device)
                kwargs = self._active_module_kwargs(batch, router_iter)
                _drop_lm_metadata(batch)
                with autocast_context(self.runtime):
                    outputs = self.model(**batch, **kwargs)
                losses.append(_loss_value(outputs["loss"]))
                if batch_index >= self.config.eval_batches:
                    break
        eval_loss = sum(losses) / len(losses) if losses else float("nan")
        return {
            "eval_loss_mean": eval_loss,
            "eval_examples": len(self.eval_lessons),
            "finite": math.isfinite(eval_loss),
        }

    def save_checkpoint(self, step: int) -> ArtifactRecord | None:
        """Save and register a tiny model checkpoint."""

        if self.model is None:
            return None
        checkpoint = self.checkpoint_manager.save_torch_checkpoint(
            self.model,
            run_id=self.run_id,
            model_type=self.config.model_type,
            module=_module_label(
                self.config.trainable_target_modules or self.config.target_modules
            ),
            step=step,
            metadata={
                "global_step": self.state.global_step,
                "routing_mode": self.config.routing_mode,
                "config": self.config.to_dict(),
                "best_eval_loss": self.state.best_eval_loss,
                "latest_train_loss": self.state.latest_train_loss,
                "latest_eval_loss": self.state.latest_eval_loss,
                "trainable_policy": (
                    self.trainable_policy.to_dict()
                    if self.trainable_policy is not None
                    else None
                ),
                "parameter_counts": dict(self.parameter_counts),
                "parameter_group_summaries": [
                    dict(item) for item in self.parameter_group_summaries
                ],
                "adapter_metadata": dict(self.adapter_metadata),
                "generated_metadata": dict(self.generated_metadata),
                "tokenizer_spec": self.tokenizer_spec.to_dict(),
                "runtime": runtime_metadata(self.runtime),
            },
        )
        self.artifacts.setdefault("checkpoint_artifact_ids", []).append(checkpoint.artifact_id)
        self.artifacts.setdefault("checkpoint_paths", []).append(checkpoint.path)
        return checkpoint

    def save_full_checkpoint(
        self,
        step: int,
        *,
        reason: str = "final",
    ) -> ArtifactRecord | None:
        """Save and register a full resumable training checkpoint."""

        if self.model is None:
            return None
        rng_state = (
            capture_rng_state()
            if self.config.save_rng_state
            else {"disabled": True, "has_python": False, "has_numpy": False, "has_torch": False, "has_cuda": False}
        )
        metadata = {
            "source": "TinyTrainer",
            "reason": reason,
            "run_id": self.run_id,
            "training_kind": self.config.training_kind,
            "model_type": self.config.model_type,
            "global_step": self.state.global_step,
            "routing_mode": self.config.routing_mode,
            "config": self.config.to_dict(),
            "source_config_kind": self.config.source_config_kind,
            "source_config": dict(self.config.source_config or {}),
            "best_eval_loss": self.state.best_eval_loss,
            "latest_train_loss": self.state.latest_train_loss,
            "latest_eval_loss": self.state.latest_eval_loss,
            "trainable_policy": (
                self.trainable_policy.to_dict()
                if self.trainable_policy is not None
                else None
            ),
            "parameter_counts": dict(self.parameter_counts),
            "parameter_group_summaries": [
                dict(item) for item in self.parameter_group_summaries
            ],
            "adapter_metadata": dict(self.adapter_metadata),
            "generated_metadata": dict(self.generated_metadata),
            "tokenizer_spec": self.tokenizer_spec.to_dict(),
            "resume_metadata": dict(self.resume_metadata),
            "runtime": runtime_metadata(self.runtime),
        }
        checkpoint = self.checkpoint_manager.save_full_training_checkpoint(
            self.model,
            optimizer=self.optimizer if self.config.save_optimizer_state else None,
            scheduler=self.scheduler if self.config.save_scheduler_state else None,
            trainer_state=self.state.to_dict(),
            config=self.config.to_dict(),
            tokenizer_spec=self.tokenizer_spec.to_dict(),
            parameter_policy=(
                self.trainable_policy.to_dict()
                if self.trainable_policy is not None
                else None
            ),
            adapter_metadata=self.adapter_metadata,
            generated_metadata=self.generated_metadata,
            rng_state=rng_state,
            run_id=self.run_id,
            model_type=self.config.model_type,
            training_kind=self.config.training_kind,
            module=_module_label(
                self.config.trainable_target_modules or self.config.target_modules
            ),
            step=step,
            metadata=metadata,
        )
        self._full_checkpoint_steps.add(step)
        self.state.full_checkpoint_artifacts.append(checkpoint.artifact_id)
        self.artifacts.setdefault("full_checkpoint_artifact_ids", []).append(
            checkpoint.artifact_id
        )
        self.artifacts.setdefault("full_checkpoint_paths", []).append(checkpoint.path)
        return checkpoint

    def load_checkpoint(self, artifact_or_path: str) -> dict:
        """Load a model checkpoint by artifact ID or path and update state metadata."""

        if self.model is None:
            raise RuntimeError("Trainer must be set up before load_checkpoint().")
        artifact = self.artifact_manager.get(artifact_or_path)
        payload = self.checkpoint_manager.load_torch_checkpoint(
            self.model,
            artifact if artifact is not None else artifact_or_path,
            map_location=str(self.device or self.config.device),
        )
        metadata = dict(payload.get("metadata", {}) or {})
        if "global_step" in metadata:
            self.state.global_step = int(metadata["global_step"])
        if metadata.get("best_eval_loss") is not None:
            self.state.best_eval_loss = float(metadata["best_eval_loss"])
        if metadata.get("latest_train_loss") is not None:
            self.state.latest_train_loss = float(metadata["latest_train_loss"])
        if metadata.get("latest_eval_loss") is not None:
            self.state.latest_eval_loss = float(metadata["latest_eval_loss"])
        return payload

    def load_full_checkpoint(self, artifact_or_path: str) -> dict[str, Any]:
        """Load a full checkpoint and restore model, optimizer, state, and RNG."""

        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer must be set up before load_full_checkpoint().")
        artifact = self.artifact_manager.get(artifact_or_path)
        checkpoint_path = artifact.path if artifact is not None else artifact_or_path
        payload = self.checkpoint_manager.load_full_training_checkpoint(
            artifact if artifact is not None else checkpoint_path,
            map_location=str(self.device or self.config.device),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        resume_info: dict[str, Any] = {
            "resumed_from_checkpoint": str(checkpoint_path),
            "resumed_from_run_id": payload.get("run_id"),
            "resume_global_step": int(payload.get("global_step", 0)),
            "checkpoint_format_version": payload.get("format_version"),
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "rng_state_restored": False,
            "load_errors": [],
        }
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
                resume_info["optimizer_state_restored"] = True
            except Exception as exc:
                resume_info["load_errors"].append(f"optimizer_state: {exc}")
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
                resume_info["scheduler_state_restored"] = True
            except Exception as exc:
                resume_info["load_errors"].append(f"scheduler_state: {exc}")
        trainer_state = payload.get("trainer_state")
        if isinstance(trainer_state, dict) and trainer_state:
            self.state = TrainerState.from_dict(trainer_state)
        else:
            self.state.global_step = int(payload.get("global_step", 0))
        if self.config.save_rng_state and isinstance(payload.get("rng_state"), dict):
            try:
                restore_rng_state(payload["rng_state"])
                resume_info["rng_state_restored"] = not payload["rng_state"].get(
                    "disabled",
                    False,
                )
            except Exception as exc:
                resume_info["load_errors"].append(f"rng_state: {exc}")
        self.resume_metadata = resume_info
        self.state.resume_metadata = dict(resume_info)
        self.artifacts["resumed_from_checkpoint"] = str(checkpoint_path)
        self.artifacts["resumed_from_run_id"] = str(payload.get("run_id"))
        return payload

    def _train_step(self) -> float:
        if self.model is None or self.optimizer is None or self._train_iter is None:
            raise RuntimeError("Trainer must be set up before training.")
        torch = _require_torch()
        self.model.train()
        if self.router is not None:
            self.router.train()
        batch = _move_lm_batch(next(self._train_iter), self.device)
        kwargs: dict[str, Any] = {}
        target_modules_for_adapters = batch.get("target_modules")
        target_modules_for_conditions = target_modules_for_adapters
        router_loss = None
        if self.config.model_type == "mop_oracle":
            active_modules = batch.pop("target_modules")
            target_modules_for_adapters = active_modules
            target_modules_for_conditions = active_modules
            kwargs["active_modules"] = active_modules
        elif self.config.model_type == "mop_learned_router":
            if self._router_train_iter is None or self.router is None:
                raise RuntimeError("learned-router trainer is missing router state.")
            router_batch = _move_router_batch(next(self._router_train_iter), self.device)
            with autocast_context(self.runtime):
                router_outputs = self.router(
                    router_batch["input_ids"],
                    attention_mask=router_batch["attention_mask"],
                    module_mask=router_batch["module_mask"],
                )
                router_loss = router_outputs["loss"]
                kwargs["active_modules"] = route_batch_with_router(
                    self.router,
                    router_batch,
                    DEFAULT_KNOWN_MODULES,
                )
            target_modules_for_conditions = target_modules_for_adapters
            batch.pop("target_modules", None)
        else:
            target_modules_for_adapters = batch.pop("target_modules", None)
            target_modules_for_conditions = target_modules_for_adapters
        kwargs.update(self._active_adapter_kwargs(target_modules_for_adapters))
        kwargs.update(self._active_condition_kwargs(target_modules_for_conditions))
        _drop_lm_metadata(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.runtime):
            outputs = self.model(**batch, **kwargs)
            loss = outputs["loss"]
            if router_loss is not None:
                loss = loss + router_loss
        loss.backward()
        self.optimizer.step()
        return _loss_value(loss)

    def _active_module_kwargs(self, batch: dict[str, Any], router_iter) -> dict[str, Any]:
        target_modules_for_adapters = batch.get("target_modules")
        target_modules_for_conditions = target_modules_for_adapters
        if self.config.model_type == "mop_oracle":
            active_modules = batch.pop("target_modules")
            kwargs = {"active_modules": active_modules}
            kwargs.update(self._active_adapter_kwargs(active_modules))
            kwargs.update(self._active_condition_kwargs(active_modules))
            return kwargs
        target_modules_for_adapters = batch.pop("target_modules", None)
        target_modules_for_conditions = target_modules_for_adapters
        if self.config.model_type == "mop_learned_router":
            if router_iter is None or self.router is None:
                raise RuntimeError("learned-router evaluation is missing router state.")
            router_batch = _move_router_batch(next(router_iter), self.device)
            with autocast_context(self.runtime):
                active_modules = route_batch_with_router(
                    self.router,
                    router_batch,
                    DEFAULT_KNOWN_MODULES,
                )
            kwargs = {"active_modules": active_modules}
            kwargs.update(self._active_adapter_kwargs(target_modules_for_adapters))
            kwargs.update(self._active_condition_kwargs(target_modules_for_conditions))
            return kwargs
        kwargs = self._active_adapter_kwargs(target_modules_for_adapters)
        kwargs.update(self._active_condition_kwargs(target_modules_for_conditions))
        return kwargs

    def _final_metrics(self) -> dict[str, Any]:
        finite = all(
            math.isfinite(value)
            for value in [
                self.state.latest_train_loss,
                self.state.latest_eval_loss,
            ]
            if value is not None
        )
        metrics = {
            "train_loss_last": self.state.latest_train_loss,
            "eval_loss_mean": self.state.latest_eval_loss,
            "best_eval_loss": self.state.best_eval_loss,
            "finite": finite,
            "global_step": self.state.global_step,
            "train_examples": len(self.train_lessons),
            "eval_examples": len(self.eval_lessons),
            "curriculum_total": self.plan.total if self.plan is not None else 0,
            "checkpoint_count": len(self.state.checkpoint_artifacts),
            "full_checkpoint_count": len(self.state.full_checkpoint_artifacts),
            "full_checkpoint_artifact_ids": list(self.state.full_checkpoint_artifacts),
            "resumed_from_checkpoint": self.resume_metadata.get(
                "resumed_from_checkpoint"
            ),
            "resume_global_step": self.resume_metadata.get("resume_global_step"),
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "queue_metadata": dict(self.queue_metadata),
            "trainable_policy": (
                self.trainable_policy.to_dict()
                if self.trainable_policy is not None
                else None
            ),
            "parameter_counts": dict(self.parameter_counts),
            "parameter_group_summaries": [
                dict(item) for item in self.parameter_group_summaries
            ],
            "adapter_metadata": dict(self.adapter_metadata),
            "generated_metadata": dict(self.generated_metadata),
            "tokenizer_spec": self.tokenizer_spec.to_dict(),
            "resume_metadata": dict(self.resume_metadata),
            "model_ref": self.config.model_ref,
            "dataset_ref": self.config.dataset_ref,
            "dataset_version_id": self.config.dataset_version_id,
            "dataset_split": self.config.dataset_split,
            "runtime": runtime_metadata(self.runtime),
        }
        return metrics

    def _generation_eval_metrics(self) -> dict[str, Any]:
        lessons = self.eval_lessons[: self.config.generation_eval_examples]
        results = []
        for lesson in lessons:
            active_modules = list(lesson.target_modules) if self.config.model_type == "mop_oracle" else None
            active_adapters = None
            active_conditions = None
            if self.config.use_fast_adapters:
                if self.config.adapter_from_target_modules:
                    active_adapters = adapter_names_from_target_modules(
                        list(lesson.target_modules)
                    )
                else:
                    active_adapters = list(self.config.active_adapters or [])
            if self.config.use_generated_params:
                if self.config.conditions_from_target_modules:
                    active_conditions = condition_names_from_target_modules(
                        list(lesson.target_modules)
                    )
                else:
                    active_conditions = list(self.config.active_conditions or [])
            results.append(
                evaluate_generated_code_for_lesson(
                    self.model,
                    self.tokenizer,
                    lesson,
                    max_new_tokens=self.config.max_new_tokens,
                    device=str(self.device or "cpu"),
                    active_modules=active_modules,
                    active_adapters=active_adapters,
                    active_conditions=active_conditions,
                )
            )
        return summarize_generation_results(results)

    def _queue_metadata(self) -> dict[str, Any]:
        if not self.config.queue_path:
            return {}
        path = Path(self.config.queue_path)
        if not path.exists():
            return {"queue_path": str(path), "exists": False}
        store = TrainingQueueStore(path)
        return {
            "queue_path": str(path),
            "exists": True,
            "counts_by_status": store.counts_by_status(),
            "counts_by_module": store.counts_by_module(),
        }

    def _apply_parameter_policy(self) -> None:
        if self.model is None:
            raise RuntimeError("Model was not created.")
        self.trainable_policy = TrainableParameterPolicy(
            mode=self.config.trainable_policy_mode,
            target_modules=self.config.trainable_target_modules,
            train_router=self.config.train_router,
            train_embeddings=self.config.train_embeddings,
            train_lm_head=self.config.train_lm_head,
            train_shared_core=self.config.train_shared_core,
            train_fast_adapters=self.config.train_fast_adapters,
            train_generated_params=self.config.train_generated_params,
            metadata={"source": "TrainerConfig"},
        )

        allow_model_empty = (
            self.trainable_policy.mode == "router_only" and self.router is not None
        )
        summaries = apply_trainable_policy(
            self.model,
            self.trainable_policy,
            allow_empty=allow_model_empty,
        )
        if self.router is not None:
            router_summaries = apply_trainable_policy(
                self.router,
                self.trainable_policy,
                allow_empty=True,
            )
            summaries = _combine_group_summaries([*summaries, *router_summaries])

        self.parameter_group_summaries = [
            summary.to_dict() for summary in summaries
        ]
        self.parameter_counts = _count_trainable_objects(self.model, self.router)
        if (
            self.parameter_counts["trainable"] == 0
            and self.trainable_policy.mode != "frozen"
        ):
            raise ValueError(
                f"Policy {self.trainable_policy.mode!r} selected zero trainable "
                "parameters across model and router."
            )
        self.state.parameter_counts = dict(self.parameter_counts)
        self.state.parameter_group_summaries = [
            dict(item) for item in self.parameter_group_summaries
        ]

    def _trainable_parameters(self) -> list[Any]:
        objects = [self.model]
        if self.router is not None:
            objects.append(self.router)
        parameters = []
        for module in objects:
            if module is None:
                continue
            parameters.extend(
                parameter for parameter in module.parameters() if parameter.requires_grad
            )
        if not parameters:
            raise ValueError("No trainable parameters are available for the optimizer.")
        return parameters

    def _active_adapter_kwargs(self, target_modules) -> dict[str, Any]:
        if not self.config.use_fast_adapters:
            return {}
        if self.config.adapter_from_target_modules:
            return {
                "active_adapters": _adapter_names_from_batch_targets(target_modules)
            }
        return {"active_adapters": list(self.config.active_adapters or [])}

    def _active_condition_kwargs(self, target_modules) -> dict[str, Any]:
        if not self.config.use_generated_params:
            return {}
        if self.config.conditions_from_target_modules:
            return {
                "active_conditions": _condition_names_from_batch_targets(target_modules)
            }
        return {"active_conditions": list(self.config.active_conditions or [])}

    def _adapter_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.use_fast_adapters),
            "adapter_names": list(self.config.fast_adapter_names or []),
            "bottleneck_dim": self.config.fast_adapter_bottleneck_dim,
            "active_adapter_mode": (
                "target_modules"
                if self.config.adapter_from_target_modules
                else "static"
            ),
            "active_adapters": list(self.config.active_adapters or []),
        }

    def _generated_metadata(self) -> dict[str, Any]:
        counts = {}
        generated_adapter = getattr(self.model, "generated_adapter", None)
        if generated_adapter is not None and hasattr(generated_adapter, "generated_parameter_count"):
            counts = generated_adapter.generated_parameter_count()
        return {
            "enabled": bool(self.config.use_generated_params),
            "condition_names": list(self.config.generated_condition_names or []),
            "condition_dim": self.config.generated_condition_dim,
            "rank": self.config.generated_rank,
            "generator_type": self.config.generated_type,
            "active_condition_mode": (
                "target_modules"
                if self.config.conditions_from_target_modules
                else "static"
            ),
            "active_conditions": list(self.config.active_conditions or []),
            "parameter_counts": dict(counts),
        }

    def _write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _write_tokenizer_spec_artifact(self) -> None:
        if "tokenizer_spec_json" in self.artifacts:
            return
        path = self.tokenizer_spec.save_json(self.run_dir / "tokenizer_spec.json")
        self.artifacts["tokenizer_spec_json"] = str(path)
        self._register_artifact(
            ArtifactRecord(
                artifact_id=f"trainer-tokenizer-{self.run_id}",
                kind="config",
                path=str(path),
                run_id=self.run_id,
                model_type=self.config.model_type,
                metadata={"source": "TokenizerSpec"},
            )
        )

    def _register_artifact(self, record: ArtifactRecord) -> None:
        if not self.artifact_manager.exists(record.artifact_id):
            self.artifact_manager.register(record)


def _lm_loader(lessons: list[KnowledgeLesson], tokenizer: TokenizerProtocol, config: TrainerConfig):
    from torch.utils.data import DataLoader

    return DataLoader(
        LessonCausalLMDataset(lessons, tokenizer, max_length=config.max_seq_len),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )


def _router_loader(lessons: list[KnowledgeLesson], tokenizer: TokenizerProtocol, config: TrainerConfig):
    from torch.utils.data import DataLoader

    return DataLoader(
        RouterDataset(
            lessons,
            tokenizer,
            known_modules=list(DEFAULT_KNOWN_MODULES),
            max_length=config.max_seq_len,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=RouterCollator(tokenizer),
    )


def _split_for_trainer(lessons: list[KnowledgeLesson], seed: int) -> tuple[list[KnowledgeLesson], list[KnowledgeLesson]]:
    if len(lessons) < 2:
        return list(lessons), list(lessons)
    train_lessons, eval_lessons = split_lessons(lessons, seed=seed)
    return train_lessons, eval_lessons or train_lessons[:]


def _combine_group_summaries(
    summaries: list[ParameterGroupSummary],
) -> list[ParameterGroupSummary]:
    grouped: dict[str, dict[str, int]] = {}
    for summary in summaries:
        entry = grouped.setdefault(
            summary.name,
            {"total": 0, "trainable": 0, "frozen": 0},
        )
        entry["total"] += int(summary.total_params)
        entry["trainable"] += int(summary.trainable_params)
        entry["frozen"] += int(summary.frozen_params)
    return [
        ParameterGroupSummary(
            name=name,
            total_params=values["total"],
            trainable_params=values["trainable"],
            frozen_params=values["frozen"],
        )
        for name, values in sorted(grouped.items())
    ]


def _count_trainable_objects(*modules) -> dict[str, int]:
    total = 0
    trainable = 0
    for module in modules:
        if module is None:
            continue
        counts = count_parameters(module)
        total += counts["total"]
        trainable += counts["trainable"]
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _spec_from_tokenizer(
    tokenizer: TokenizerProtocol,
    fallback: TokenizerSpec,
) -> TokenizerSpec:
    to_spec = getattr(tokenizer, "to_spec", None)
    if callable(to_spec):
        return to_spec()
    return fallback


def _adapter_names_from_batch_targets(target_modules) -> list[str] | list[list[str]]:
    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        return adapter_names_from_target_modules([target_modules])
    modules_list = list(target_modules)
    if not modules_list:
        return []
    if all(isinstance(module, str) for module in modules_list):
        return adapter_names_from_target_modules(modules_list)
    return [
        adapter_names_from_target_modules(list(module_names or []))
        for module_names in modules_list
    ]


def _condition_names_from_batch_targets(target_modules) -> list[str] | list[list[str]]:
    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        return condition_names_from_target_modules([target_modules])
    modules_list = list(target_modules)
    if not modules_list:
        return []
    if all(isinstance(module, str) for module in modules_list):
        return condition_names_from_target_modules(modules_list)
    return [
        condition_names_from_target_modules(list(module_names or []))
        for module_names in modules_list
    ]


def _move_lm_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    return move_batch_to_device(dict(batch), str(device))


def _move_router_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    return move_batch_to_device(dict(batch), str(device))


def _drop_lm_metadata(batch: dict[str, Any]) -> None:
    for key in ("lesson_id", "metadata", "domain", "skill"):
        batch.pop(key, None)


def _loss_value(loss: Any) -> float:
    if loss is None:
        return float("nan")
    return float(loss.detach().cpu().item())


def _module_label(target_modules: list[str] | None) -> str:
    if not target_modules:
        return "all"
    return "-".join(target_modules)


def _runtime_config_from_trainer(config: TrainerConfig) -> RuntimeConfig:
    return RuntimeConfig(
        device=config.device,
        precision=config.precision,
        enable_amp=bool(config.enable_amp or config.use_amp),
        allow_tf32=config.allow_tf32,
        deterministic=config.deterministic,
        compile_model=config.compile_model,
        require_device_available=config.require_device_available,
    )


def _make_run_id(run_name: str) -> str:
    safe_name = "".join(ch.lower() if ch.isalnum() else "-" for ch in run_name).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name or 'trainer'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for TinyTrainer training.") from exc
    return torch


def _require_collators() -> None:
    if CausalLMCollator is None:
        raise RuntimeError("PyTorch is required for CausalLMCollator.")
    if RouterCollator is None:
        raise RuntimeError("PyTorch is required for RouterCollator.")
    if TinyCausalTransformer is None or TinyMoPCausalTransformer is None or TinyModuleRouter is None:
        raise RuntimeError("PyTorch is required for tiny trainer models.")
