"""Tiny continued-pretraining runner over text corpus chunks."""

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
from mopforge.experiments.utils import set_seed, split_lessons
from mopforge.kts import LessonStore
from mopforge.lifecycle import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
    restore_rng_state,
)
from mopforge.models import TinyCausalTransformer, TinyMoPCausalTransformer
from mopforge.pretrain.config import ContinuedPretrainConfig, ContinuedPretrainResult
from mopforge.pretrain.corpus import (
    TextCorpusStore,
    build_corpus_from_lessons,
)
from mopforge.pretrain.dataset import CorpusCausalLMCollator, CorpusCausalLMDataset
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
    build_tokenizer,
    get_tokenizer_vocab_size,
    tokenizer_spec_from_config,
)
from mopforge.training import DEFAULT_KNOWN_MODULES
from mopforge.training.parameter_policy import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    count_parameters,
)


def run_continued_pretraining(
    config: ContinuedPretrainConfig,
) -> ContinuedPretrainResult:
    """Run one CPU-smoke continued-pretraining loop."""

    config = ContinuedPretrainConfig(**config.to_dict())
    torch = _require_torch()
    if CorpusCausalLMCollator is None:
        raise RuntimeError("PyTorch is required for CorpusCausalLMCollator.")
    _require_models()
    set_seed(config.seed)
    runtime = build_runtime_context(_runtime_config_from_pretrain(config))
    apply_runtime_determinism(runtime, config.seed)

    run_id = _make_run_id(config.run_name)
    registry = RunRegistry(config.run_registry_root)
    run_dir = registry.create_run_dir(run_id)
    artifact_manager = ArtifactManager(config.artifact_root)
    checkpoint_manager = CheckpointManager(artifact_manager)
    tokenizer_spec = tokenizer_spec_from_config(config)
    tokenizer = build_tokenizer(tokenizer_spec)
    tokenizer_spec = _spec_from_tokenizer(tokenizer, tokenizer_spec)
    tokenizer_spec_path = tokenizer_spec.save_json(run_dir / "tokenizer_spec.json")
    device = torch.device(runtime.device_info.selected)

    store = TextCorpusStore(config.corpus_path)
    records = store.load_all()
    if not records:
        records = _build_records_from_lessons(config)
        if records:
            store.add_many(records)
    if not records:
        raise ValueError("Continued pretraining corpus is empty.")

    dataset = CorpusCausalLMDataset(
        records,
        tokenizer,
        max_seq_len=config.max_seq_len,
        stride=config.stride,
    )
    if len(dataset) == 0:
        raise ValueError("Continued pretraining corpus produced no chunks.")

    train_dataset, eval_dataset = _split_dataset(dataset, config.seed)
    train_loader = _loader(train_dataset, tokenizer, config)
    eval_loader = _loader(eval_dataset, tokenizer, config)
    train_iter = cycle(train_loader)

    model = move_model_to_runtime(_build_model(config, tokenizer), runtime)
    policy = TrainableParameterPolicy(
        mode=config.trainable_policy_mode,
        train_fast_adapters=config.use_fast_adapters,
        train_generated_params=config.use_generated_params,
        metadata={"source": "ContinuedPretrainConfig"},
    )
    summaries = apply_trainable_policy(model, policy)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    global_step = 0
    resume_metadata: dict[str, Any] = {}
    full_checkpoint_artifacts: list[str] = []
    full_checkpoint_paths: list[str] = []
    full_checkpoint_steps: set[int] = set()
    if config.resume_from_checkpoint:
        resume_artifact = artifact_manager.get(config.resume_from_checkpoint)
        payload = checkpoint_manager.load_full_training_checkpoint(
            resume_artifact if resume_artifact is not None else config.resume_from_checkpoint,
            map_location=str(device),
        )
        model.load_state_dict(payload["model_state_dict"])
        global_step = int(payload.get("global_step", 0))
        resume_metadata = {
            "resumed_from_checkpoint": (
                resume_artifact.path if resume_artifact is not None else config.resume_from_checkpoint
            ),
            "resumed_from_run_id": payload.get("run_id"),
            "resume_global_step": global_step,
            "checkpoint_format_version": payload.get("format_version"),
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "load_errors": [],
        }
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                resume_metadata["optimizer_state_restored"] = True
            except Exception as exc:
                resume_metadata["load_errors"].append(f"optimizer_state: {exc}")
        if config.save_rng_state and isinstance(payload.get("rng_state"), dict):
            try:
                restore_rng_state(payload["rng_state"])
                resume_metadata["rng_state_restored"] = not payload["rng_state"].get(
                    "disabled",
                    False,
                )
            except Exception as exc:
                resume_metadata["load_errors"].append(f"rng_state: {exc}")

    latest_train_loss = None
    while global_step < config.max_steps:
        latest_train_loss = _train_step(
            model,
            optimizer,
            next(train_iter),
            device,
            config,
            runtime,
        )
        global_step += 1
        if (
            config.save_full_checkpoints
            and config.checkpoint_every_steps is not None
            and global_step % config.checkpoint_every_steps == 0
        ):
            checkpoint = _save_full_pretrain_checkpoint(
                checkpoint_manager=checkpoint_manager,
                model=model,
                optimizer=optimizer,
                config=config,
                run_id=run_id,
                global_step=global_step,
                tokenizer_spec=tokenizer_spec,
                policy=policy,
                parameter_counts=count_parameters(model),
                parameter_group_summaries=[summary.to_dict() for summary in summaries],
                adapter_metadata=_adapter_metadata(model, config),
                generated_metadata=_generated_metadata(model, config),
                latest_train_loss=latest_train_loss,
                latest_eval_loss=None,
                resume_metadata=resume_metadata,
                runtime=runtime,
                reason="interval",
            )
            full_checkpoint_artifacts.append(checkpoint.artifact_id)
            full_checkpoint_paths.append(checkpoint.path)
            full_checkpoint_steps.add(global_step)
    latest_eval_loss = _evaluate(model, eval_loader, device, config, runtime)
    finite = all(
        math.isfinite(value)
        for value in [latest_train_loss, latest_eval_loss]
        if value is not None
    )

    parameter_counts = count_parameters(model)
    summary_dicts = [summary.to_dict() for summary in summaries]
    corpus_summary = {
        "corpus_path": str(config.corpus_path),
        "records": len(records),
        "chunks": len(dataset),
        "max_seq_len": config.max_seq_len,
        "stride": config.stride or config.max_seq_len,
        "sources": _count_by(records, "source"),
        "domains": _count_by(records, "domain"),
        "languages": _count_by(records, "language"),
    }
    corpus_summary_path = _write_json(run_dir / "corpus_summary.json", corpus_summary)

    checkpoint_artifact = None
    if config.save_checkpoints:
        checkpoint_artifact = checkpoint_manager.save_torch_checkpoint(
            model,
            run_id=run_id,
            model_type=config.model_type,
            module="continued_pretraining",
            step=global_step,
            metadata={
                "config": config.to_dict(),
                "final_train_loss": latest_train_loss,
                "final_eval_loss": latest_eval_loss,
                "parameter_counts": parameter_counts,
                "parameter_group_summaries": summary_dicts,
                "tokenizer_spec": tokenizer_spec.to_dict(),
                "adapter_metadata": _adapter_metadata(model, config),
                "generated_metadata": _generated_metadata(model, config),
                "runtime": runtime_metadata(runtime),
            },
        )

    if config.save_full_checkpoints and global_step not in full_checkpoint_steps:
        checkpoint = _save_full_pretrain_checkpoint(
            checkpoint_manager=checkpoint_manager,
            model=model,
            optimizer=optimizer,
            config=config,
            run_id=run_id,
            global_step=global_step,
            tokenizer_spec=tokenizer_spec,
            policy=policy,
            parameter_counts=parameter_counts,
            parameter_group_summaries=summary_dicts,
            adapter_metadata=_adapter_metadata(model, config),
            generated_metadata=_generated_metadata(model, config),
            latest_train_loss=latest_train_loss,
            latest_eval_loss=latest_eval_loss,
            resume_metadata=resume_metadata,
            runtime=runtime,
            reason="final",
        )
        full_checkpoint_artifacts.append(checkpoint.artifact_id)
        full_checkpoint_paths.append(checkpoint.path)

    metrics: dict[str, Any] = {
        "train_loss_last": latest_train_loss,
        "eval_loss_mean": latest_eval_loss,
        "finite": finite,
        "global_step": global_step,
        "corpus_records": len(records),
        "corpus_chunks": len(dataset),
        "parameter_counts": parameter_counts,
        "parameter_group_summaries": summary_dicts,
        "trainable_policy": policy.to_dict(),
        "tokenizer_spec": tokenizer_spec.to_dict(),
        "adapter_metadata": _adapter_metadata(model, config),
        "generated_metadata": _generated_metadata(model, config),
        "continued_pretraining": True,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "resumed_from_checkpoint": resume_metadata.get("resumed_from_checkpoint"),
        "resume_global_step": resume_metadata.get("resume_global_step"),
        "resume_metadata": dict(resume_metadata),
        "checkpoint_artifact_ids": (
            [checkpoint_artifact.artifact_id] if checkpoint_artifact is not None else []
        ),
        "full_checkpoint_artifact_ids": list(full_checkpoint_artifacts),
        "model_ref": config.model_ref,
        "corpus_dataset_ref": config.corpus_dataset_ref,
        "dataset_split": config.dataset_split,
        "runtime": runtime_metadata(runtime),
    }
    artifacts: dict[str, str] = {
        "corpus_summary_json": str(corpus_summary_path),
        "tokenizer_spec_json": str(tokenizer_spec_path),
    }
    if checkpoint_artifact is not None:
        artifacts["checkpoint_artifact_id"] = checkpoint_artifact.artifact_id
        artifacts["checkpoint_path"] = checkpoint_artifact.path
    if full_checkpoint_artifacts:
        artifacts["full_checkpoint_artifact_id"] = full_checkpoint_artifacts[-1]
        artifacts["full_checkpoint_path"] = full_checkpoint_paths[-1]

    run_record = TrainingRunRecord(
        run_id=run_id,
        run_name=config.run_name,
        model_type=config.model_type,
        curriculum_strategy=config.curriculum_strategy,
        started_at=_now(),
        finished_at=_now(),
        config=config.to_dict(),
        metrics=metrics,
        artifacts=dict(artifacts),
    )
    run_json_path = registry.save(run_record)
    artifacts.update(
        {
            "run_json": str(run_json_path),
            "metrics_json": str(run_dir / "metrics.json"),
        }
    )
    artifact_manager.register(
        ArtifactRecord(
            artifact_id=f"continued-pretrain-metrics-{run_id}",
            kind="metrics",
            path=artifacts["metrics_json"],
            run_id=run_id,
            model_type=config.model_type,
            metadata={"source": "run_continued_pretraining"},
        )
    )

    artifact_manager.register(
        ArtifactRecord(
            artifact_id=f"continued-pretrain-corpus-{run_id}",
            kind="config",
            path=str(corpus_summary_path),
            run_id=run_id,
            model_type=config.model_type,
            metadata={"source": "run_continued_pretraining"},
        )
    )
    artifact_manager.register(
        ArtifactRecord(
            artifact_id=f"continued-pretrain-tokenizer-{run_id}",
            kind="config",
            path=str(tokenizer_spec_path),
            run_id=run_id,
            model_type=config.model_type,
            metadata={"source": "TokenizerSpec"},
        )
    )

    result = ContinuedPretrainResult(
        run_id=run_id,
        run_name=config.run_name,
        model_type=config.model_type,
        corpus_records=len(records),
        corpus_chunks=len(dataset),
        final_train_loss=latest_train_loss,
        final_eval_loss=latest_eval_loss,
        metrics=metrics,
        artifacts=artifacts,
        finite=finite,
    )
    result_path = result.save_json(run_dir / "continued_pretrain_result.json")
    result.artifacts["continued_pretrain_result_json"] = str(result_path)
    result.save_json(result_path)
    artifact_manager.register(
        ArtifactRecord(
            artifact_id=f"continued-pretrain-result-{run_id}",
            kind="metrics",
            path=str(result_path),
            run_id=run_id,
            model_type=config.model_type,
            metadata={"source": "run_continued_pretraining"},
        )
    )
    return result


def _build_records_from_lessons(config: ContinuedPretrainConfig):
    if not config.lesson_path:
        return []
    lesson_path = Path(config.lesson_path)
    if not lesson_path.exists():
        return []
    lessons = LessonStore(lesson_path).load_all()
    return build_corpus_from_lessons(lessons)


def _split_dataset(dataset: CorpusCausalLMDataset, seed: int):
    items = [dataset[index] for index in range(len(dataset))]
    if len(items) < 2:
        return _ListDataset(items), _ListDataset(items)
    train_lessons, eval_lessons = split_lessons(items, seed=seed)  # type: ignore[arg-type]
    return _ListDataset(train_lessons), _ListDataset(eval_lessons or train_lessons[:])


class _ListDataset:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _loader(dataset, tokenizer: TokenizerProtocol, config: ContinuedPretrainConfig):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CorpusCausalLMCollator(tokenizer),
    )


def _build_model(config: ContinuedPretrainConfig, tokenizer: TokenizerProtocol):
    if config.model_type == "dense":
        return TinyCausalTransformer(
            vocab_size=get_tokenizer_vocab_size(tokenizer),
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            use_fast_adapters=config.use_fast_adapters,
            fast_adapter_names=config.fast_adapter_names,
            use_generated_params=config.use_generated_params,
            generated_condition_names=config.generated_condition_names,
            generated_condition_dim=config.generated_condition_dim,
            generated_rank=config.generated_rank,
            generated_type=config.generated_type,
        )
    return TinyMoPCausalTransformer(
        vocab_size=get_tokenizer_vocab_size(tokenizer),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        module_names=DEFAULT_KNOWN_MODULES,
        use_fast_adapters=config.use_fast_adapters,
        fast_adapter_names=config.fast_adapter_names,
        use_generated_params=config.use_generated_params,
        generated_condition_names=config.generated_condition_names,
        generated_condition_dim=config.generated_condition_dim,
        generated_rank=config.generated_rank,
        generated_type=config.generated_type,
    )


def _train_step(model, optimizer, batch: dict[str, Any], device, config: ContinuedPretrainConfig, runtime) -> float:
    model.train()
    batch = _move_batch(batch, device)
    kwargs = _model_kwargs(config)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(runtime):
        outputs = model(**batch, **kwargs)
        loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    return _loss_value(loss)


def _evaluate(model, loader, device, config: ContinuedPretrainConfig, runtime) -> float:
    torch = _require_torch()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            with autocast_context(runtime):
                outputs = model(**batch, **_model_kwargs(config))
            losses.append(_loss_value(outputs["loss"]))
            if batch_index >= config.eval_batches:
                break
    return sum(losses) / len(losses) if losses else float("nan")


def _model_kwargs(config: ContinuedPretrainConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.use_fast_adapters:
        kwargs["active_adapters"] = list(config.fast_adapter_names or [])
    if config.use_generated_params:
        kwargs["active_conditions"] = list(config.generated_condition_names or [])
    return kwargs


def _save_full_pretrain_checkpoint(
    *,
    checkpoint_manager: CheckpointManager,
    model,
    optimizer,
    config: ContinuedPretrainConfig,
    run_id: str,
    global_step: int,
    tokenizer_spec,
    policy: TrainableParameterPolicy,
    parameter_counts: dict[str, int],
    parameter_group_summaries: list[dict[str, Any]],
    adapter_metadata: dict[str, Any],
    generated_metadata: dict[str, Any],
    latest_train_loss: float | None,
    latest_eval_loss: float | None,
    resume_metadata: dict[str, Any],
    runtime,
    reason: str,
) -> ArtifactRecord:
    rng_state = (
        capture_rng_state()
        if config.save_rng_state
        else {"disabled": True, "has_python": False, "has_numpy": False, "has_torch": False, "has_cuda": False}
    )
    trainer_state = {
        "global_step": global_step,
        "epoch": 0,
        "latest_train_loss": latest_train_loss,
        "latest_eval_loss": latest_eval_loss,
        "parameter_counts": dict(parameter_counts),
        "parameter_group_summaries": [
            dict(item) for item in parameter_group_summaries
        ],
        "resume_metadata": dict(resume_metadata),
    }
    metadata = {
        "source": "run_continued_pretraining",
        "reason": reason,
        "run_id": run_id,
        "training_kind": "pretrain",
        "model_type": config.model_type,
        "global_step": global_step,
        "config": config.to_dict(),
        "latest_train_loss": latest_train_loss,
        "latest_eval_loss": latest_eval_loss,
        "parameter_counts": dict(parameter_counts),
        "parameter_group_summaries": [
            dict(item) for item in parameter_group_summaries
        ],
        "adapter_metadata": dict(adapter_metadata),
        "generated_metadata": dict(generated_metadata),
        "tokenizer_spec": tokenizer_spec.to_dict(),
        "trainable_policy": policy.to_dict(),
        "resume_metadata": dict(resume_metadata),
        "runtime": runtime_metadata(runtime),
    }
    return checkpoint_manager.save_full_training_checkpoint(
        model,
        optimizer=optimizer if config.save_optimizer_state else None,
        scheduler=None,
        trainer_state=trainer_state,
        config=config.to_dict(),
        tokenizer_spec=tokenizer_spec.to_dict(),
        parameter_policy=policy.to_dict(),
        adapter_metadata=adapter_metadata,
        generated_metadata=generated_metadata,
        rng_state=rng_state,
        run_id=run_id,
        model_type=config.model_type,
        training_kind="pretrain",
        module="continued_pretraining",
        step=global_step,
        metadata=metadata,
    )


def _move_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    moved = move_batch_to_device(dict(batch), str(device))
    for key in ("record_id", "chunk_index"):
        moved.pop(key, None)
    return moved


def _loss_value(loss: Any) -> float:
    if loss is None:
        return float("nan")
    return float(loss.detach().cpu().item())


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _spec_from_tokenizer(tokenizer: TokenizerProtocol, fallback):
    to_spec = getattr(tokenizer, "to_spec", None)
    if callable(to_spec):
        return to_spec()
    return fallback


def _generated_metadata(model, config: ContinuedPretrainConfig) -> dict[str, Any]:
    counts = {}
    generated_adapter = getattr(model, "generated_adapter", None)
    if generated_adapter is not None and hasattr(generated_adapter, "generated_parameter_count"):
        counts = generated_adapter.generated_parameter_count()
    return {
        "enabled": bool(config.use_generated_params),
        "condition_names": list(config.generated_condition_names or []),
        "condition_dim": config.generated_condition_dim,
        "rank": config.generated_rank,
        "generator_type": config.generated_type,
        "active_condition_mode": "static",
        "active_conditions": list(config.generated_condition_names or []),
        "parameter_counts": dict(counts),
    }


def _adapter_metadata(model, config: ContinuedPretrainConfig) -> dict[str, Any]:
    return {
        "enabled": bool(config.use_fast_adapters),
        "adapter_names": list(config.fast_adapter_names or []),
        "bottleneck_dim": getattr(
            getattr(model, "fast_adapter_bank", None),
            "config",
            None,
        ).bottleneck_dim if getattr(model, "fast_adapter_bank", None) is not None else None,
        "active_adapter_mode": "static",
        "active_adapters": list(config.fast_adapter_names or []),
    }


def _count_by(records, field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = getattr(record, field_name)
        key = str(value) if value is not None else "none"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _runtime_config_from_pretrain(config: ContinuedPretrainConfig) -> RuntimeConfig:
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
    return f"{timestamp}-{safe_name or 'continued-pretrain'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for continued pretraining.") from exc
    return torch


def _require_models() -> None:
    if TinyCausalTransformer is None or TinyMoPCausalTransformer is None:
        raise RuntimeError("PyTorch is required for tiny continued-pretraining models.")
