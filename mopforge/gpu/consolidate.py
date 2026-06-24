"""Consolidate a distributed sharded checkpoint for evaluation and export."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.gpu.checkpointing import save_gpu_checkpoint
from mopforge.gpu.config import GPUTrainingConfig
from mopforge.gpu.distributed import DistributedRuntime
from mopforge.gpu.distributed_checkpoint import load_sharded_training_checkpoint
from mopforge.models import (
    ModelArchitectureConfig,
    architecture_from_gpu_config,
    build_tiny_model_from_architecture,
)
from mopforge.tokenization import build_tokenizer, tokenizer_spec_from_config
from mopforge.training.parameter_policy import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    build_optimizer_for_trainable_parameters,
)


def consolidate_sharded_gpu_checkpoint(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict:
    """Load DCP model/optimizer shards into CPU and write one model checkpoint."""

    torch = _require_torch()
    source = Path(checkpoint_dir)
    sidecar = _torch_load(source / "metadata.pt")
    if sidecar.get("checkpoint_format") != "mopforge_distributed_sharded_v1":
        raise ValueError(f"Unsupported sharded checkpoint: {source}")
    if config_path is None:
        config_data = dict(sidecar.get("config") or {})
    else:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config_data = dict(raw.get("payload") or raw)
    if not config_data:
        raise ValueError("Sharded checkpoint has no config snapshot; pass config_path.")
    config = GPUTrainingConfig.from_dict(config_data)
    tokenizer = build_tokenizer(tokenizer_spec_from_config(config))
    model_metadata = dict((sidecar.get("metadata") or {}).get("model") or {})
    architecture_data = dict(model_metadata.get("architecture") or {})
    architecture = (
        ModelArchitectureConfig.from_dict(architecture_data)
        if architecture_data
        else architecture_from_gpu_config(config)
    )
    model = build_tiny_model_from_architecture(architecture, tokenizer=tokenizer)
    policy = _policy_from_config(config)
    apply_trainable_policy(model, policy)
    optimizer = build_optimizer_for_trainable_parameters(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    restored = load_sharded_training_checkpoint(
        source,
        model=model,
        optimizer=optimizer,
        runtime=DistributedRuntime(),
    )
    metadata = dict(restored.get("metadata") or {})
    saved = save_gpu_checkpoint(
        output_path,
        model=model,
        state=restored.get("trainer_state"),
        config=config,
        runtime_metadata=metadata.get("runtime"),
        data_metadata=metadata.get("data"),
        model_metadata=metadata.get("model") or {"architecture": architecture.to_dict()},
        trainable_policy=policy.to_dict(),
    )
    result = {
        "format": "mopforge_sharded_consolidation_v1",
        "source": str(source),
        "output": saved,
        "source_world_size": (restored.get("distributed") or {}).get("world_size"),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "optimizer_state_exported": False,
        "architecture": architecture.to_dict(),
    }
    report_path = Path(saved).with_suffix(Path(saved).suffix + ".consolidation.json")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(report_path)
    del optimizer, model
    return result


def _policy_from_config(config):
    return TrainableParameterPolicy(
        mode=config.trainable_policy_mode,
        target_modules=config.target_modules or None,
        train_router=bool(config.metadata.get("train_router", False)),
        train_lm_head=bool(
            config.metadata.get("train_lm_head", False)
            or config.trainable_policy_mode == "adapters_norm_head"
        ),
        train_norm=bool(
            config.metadata.get("train_norm", False)
            or config.trainable_policy_mode == "adapters_norm_head"
        ),
        train_fast_adapters=config.use_fast_adapters,
        train_lora_deltas=config.use_lora_deltas,
        train_generated_params=config.use_generated_params,
        metadata={"training_kind": "gpu_train"},
    )


def _torch_load(path):
    torch = _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for checkpoint consolidation.") from exc
    return torch
