"""Trainable-parameter policy helpers for tiny MoP models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_POLICY_MODES = {
    "all",
    "core_only",
    "modules_only",
    "target_modules_only",
    "router_only",
    "head_only",
    "fast_adapters_only",
    "generated_params_only",
    "frozen",
}


@dataclass(slots=True)
class ParameterGroupSummary:
    """Parameter counts for one inferred parameter group."""

    name: str
    total_params: int
    trainable_params: int
    frozen_params: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dictionary."""

        return {
            "name": self.name,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
        }


@dataclass(slots=True)
class TrainableParameterPolicy:
    """Policy for freezing and unfreezing model parameter groups."""

    mode: str = "all"
    target_modules: list[str] | None = None
    train_router: bool = False
    train_embeddings: bool = False
    train_lm_head: bool = False
    train_shared_core: bool = True
    train_fast_adapters: bool = False
    train_generated_params: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate policy settings."""

        if self.mode not in SUPPORTED_POLICY_MODES:
            valid = ", ".join(sorted(SUPPORTED_POLICY_MODES))
            raise ValueError(f"mode must be one of: {valid}.")
        if self.target_modules is not None:
            if not all(isinstance(module, str) and module.strip() for module in self.target_modules):
                raise ValueError("target_modules must contain non-empty strings.")
            seen = set()
            self.target_modules = [
                module for module in self.target_modules
                if not (module in seen or seen.add(module))
            ]
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy dictionary."""

        return {
            "mode": self.mode,
            "target_modules": list(self.target_modules or []),
            "train_router": self.train_router,
            "train_embeddings": self.train_embeddings,
            "train_lm_head": self.train_lm_head,
            "train_shared_core": self.train_shared_core,
            "train_fast_adapters": self.train_fast_adapters,
            "train_generated_params": self.train_generated_params,
            "metadata": dict(self.metadata),
        }


def infer_parameter_group(name: str) -> str:
    """Infer a stable high-level parameter group from a parameter name."""

    parts = name.split(".")
    if "generated_adapter" in parts:
        if "condition_embedding" in parts:
            return "generated_condition_embedding"
        if "generator" in parts:
            return "hypernetwork"
        return "generated_params"
    if "fast_adapter_bank" in parts or "adapters" in parts:
        for index, part in enumerate(parts):
            if part == "adapters" and index + 1 < len(parts):
                return f"adapter:{parts[index + 1]}"
        return "fast_adapter"
    if "module_bank" in parts:
        for index, part in enumerate(parts):
            if part == "blocks" and index + 1 < len(parts):
                return f"module:{parts[index + 1]}"
        return "module_bank"
    if "lm_head" in parts or parts[-1].endswith("head"):
        return "lm_head"
    if "token_embedding" in parts or "position_embedding" in parts:
        return "embeddings"
    if parts[0] == "embedding":
        return "embeddings"
    if parts[0] == "router" or parts[0] == "mlp":
        return "router"
    if "shared_blocks" in parts or parts[0] == "blocks":
        return "shared_core"
    if parts[0] == "norm" or "norm" in parts:
        return "norm"
    return "other"


def summarize_parameter_groups(model) -> list[ParameterGroupSummary]:
    """Summarize total/trainable/frozen counts by inferred parameter group."""

    groups: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        group = _group_for_model_param(model, name)
        count = int(parameter.numel())
        entry = groups.setdefault(group, {"total": 0, "trainable": 0})
        entry["total"] += count
        if parameter.requires_grad:
            entry["trainable"] += count
    summaries = []
    for group_name in sorted(groups):
        total = groups[group_name]["total"]
        trainable = groups[group_name]["trainable"]
        summaries.append(
            ParameterGroupSummary(
                name=group_name,
                total_params=total,
                trainable_params=trainable,
                frozen_params=total - trainable,
            )
        )
    return summaries


def apply_trainable_policy(
    model,
    policy: TrainableParameterPolicy,
    *,
    allow_empty: bool = False,
) -> list[ParameterGroupSummary]:
    """Apply a trainable-parameter policy to a PyTorch model."""

    policy.__post_init__()
    target_modules = set(policy.target_modules or [])
    trainable_count = 0
    for name, parameter in model.named_parameters():
        group = _group_for_model_param(model, name)
        should_train = _should_train_group(group, policy, target_modules)
        parameter.requires_grad = should_train
        if should_train:
            trainable_count += int(parameter.numel())

    if trainable_count == 0 and policy.mode != "frozen" and not allow_empty:
        raise ValueError(
            f"Policy {policy.mode!r} selected zero trainable parameters."
        )
    return summarize_parameter_groups(model)


def count_parameters(model) -> dict[str, int]:
    """Return total/trainable/frozen parameter counts."""

    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_optimizer_for_trainable_parameters(
    model,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
):
    """Build AdamW over parameters with ``requires_grad=True`` only."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to build an optimizer.") from exc
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters are available for the optimizer.")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def policy_from_queue_item(item) -> TrainableParameterPolicy:
    """Create a target-module policy from a training queue item."""

    if item.module in {"generated_params", "hypernetwork"} or item.metadata.get("train_generated_params"):
        condition_name = item.metadata.get("condition_name")
        target_modules = [condition_name] if condition_name else None
        return TrainableParameterPolicy(
            mode="generated_params_only",
            target_modules=target_modules,
            train_generated_params=True,
            metadata={
                "queue_item_id": item.item_id,
                "lesson_id": item.lesson_id,
                "module": item.module,
            },
        )
    if item.module == "fast_adapter" or item.metadata.get("train_fast_adapter"):
        adapter_name = item.metadata.get("adapter_name")
        target_modules = [adapter_name] if adapter_name else None
        return TrainableParameterPolicy(
            mode="fast_adapters_only",
            target_modules=target_modules,
            train_fast_adapters=True,
            metadata={
                "queue_item_id": item.item_id,
                "lesson_id": item.lesson_id,
                "module": item.module,
            },
        )
    return TrainableParameterPolicy(
        mode="target_modules_only",
        target_modules=[item.module],
        metadata={"queue_item_id": item.item_id, "lesson_id": item.lesson_id},
    )


def _group_for_model_param(model, name: str) -> str:
    if model.__class__.__name__ == "TinyModuleRouter":
        return "router"
    return infer_parameter_group(name)


def _should_train_group(
    group: str,
    policy: TrainableParameterPolicy,
    target_modules: set[str],
) -> bool:
    generated_group = _is_generated_group(group)
    if policy.mode == "all":
        if generated_group:
            return policy.train_generated_params
        return True
    if policy.mode == "frozen":
        return False
    if policy.mode == "head_only":
        if generated_group:
            return policy.train_generated_params
        return group == "lm_head"
    if policy.mode == "generated_params_only":
        return generated_group
    if policy.mode == "fast_adapters_only":
        if generated_group:
            return policy.train_generated_params
        if group.startswith("adapter:"):
            if not target_modules:
                return True
            return group.split(":", 1)[1] in target_modules
        return group == "fast_adapter"
    if policy.mode == "router_only":
        if generated_group:
            return policy.train_generated_params
        return group == "router"
    if policy.mode == "core_only":
        if generated_group:
            return policy.train_generated_params
        if group.startswith("adapter:") or group == "fast_adapter":
            return policy.train_fast_adapters
        if group.startswith("module:") or group in {"module_bank", "router"}:
            return policy.train_router and group == "router"
        if group == "embeddings":
            return policy.train_embeddings
        if group == "lm_head":
            return policy.train_lm_head
        return policy.train_shared_core
    if policy.mode == "modules_only":
        if generated_group:
            return policy.train_generated_params
        if group.startswith("module:") or group == "module_bank":
            return True
        if group.startswith("adapter:") or group == "fast_adapter":
            return policy.train_fast_adapters
        if group == "embeddings":
            return policy.train_embeddings
        if group == "lm_head":
            return policy.train_lm_head
        if group == "router":
            return policy.train_router
        return False
    if policy.mode == "target_modules_only":
        if generated_group:
            return policy.train_generated_params
        if group.startswith("module:"):
            return group.split(":", 1)[1] in target_modules
        if group.startswith("adapter:"):
            return (
                policy.train_fast_adapters
                and group.split(":", 1)[1] in target_modules
            )
        if group == "fast_adapter":
            return policy.train_fast_adapters
        if group == "embeddings":
            return policy.train_embeddings
        if group == "lm_head":
            return policy.train_lm_head
        if group == "router":
            return policy.train_router
        return False
    return False


def _is_generated_group(group: str) -> bool:
    return group in {
        "generated_params",
        "generated_condition_embedding",
        "hypernetwork",
    }
