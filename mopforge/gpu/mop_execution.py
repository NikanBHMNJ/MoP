"""GPU-aware MoP routing and fast-parameter metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mopforge.models.fast_adapters import adapter_names_from_target_modules
from mopforge.models.generated_params import condition_names_from_target_modules
from mopforge.training.parameter_policy import count_parameters, infer_parameter_group
from mopforge.training.routing import normalize_target_modules


@dataclass(slots=True)
class ModuleRoutingPlan:
    module_names: list[str]
    batch_size: int
    active_by_module: dict[str, list[int]]
    active_by_sample: list[list[str]]
    density: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_module_routing_plan(
    active_modules,
    known_modules,
    *,
    always_include_core: bool = True,
) -> ModuleRoutingPlan:
    known = list(known_modules)
    if active_modules is None:
        per_sample = [normalize_target_modules([], known, always_include_core=always_include_core)]
    elif isinstance(active_modules, str) or all(isinstance(item, str) for item in list(active_modules)):
        per_sample = [
            normalize_target_modules(
                active_modules,
                known,
                always_include_core=always_include_core,
            )
        ]
    else:
        per_sample = [
            normalize_target_modules(
                modules,
                known,
                always_include_core=always_include_core,
            )
            for modules in active_modules
        ]
    active_by_module = {name: [] for name in known}
    for index, modules in enumerate(per_sample):
        for module in modules:
            active_by_module.setdefault(module, []).append(index)
    return ModuleRoutingPlan(
        module_names=known,
        batch_size=len(per_sample),
        active_by_module={key: value for key, value in active_by_module.items() if value},
        active_by_sample=per_sample,
        density=routing_density(per_sample, known),
        metadata={"implementation": "pytorch_grouping_metadata"},
    )


def routing_density(active_by_sample, known_modules) -> float:
    per_sample = list(active_by_sample or [])
    known_count = max(1, len(list(known_modules or [])))
    if not per_sample:
        return 0.0
    active = sum(len(set(sample)) for sample in per_sample)
    return float(active) / float(len(per_sample) * known_count)


def group_batch_by_modules(active_modules, known_modules) -> dict[tuple[str, ...], list[int]]:
    plan = build_module_routing_plan(active_modules, known_modules)
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, modules in enumerate(plan.active_by_sample):
        groups.setdefault(tuple(modules), []).append(index)
    return groups


def estimate_active_parameters(model, active_modules=None) -> dict[str, Any]:
    totals = count_parameters(model)
    module_names = list(getattr(model, "module_names", []))
    if not module_names:
        return {
            "active_params": totals["trainable"],
            "total_params": totals["total"],
            "active_ratio": 1.0,
            "active_trainable_params": totals["trainable"],
            "active_trainable_ratio": totals["trainable"] / max(1, totals["total"]),
            "shared_frozen_params": totals["frozen"],
            "routed_module_params": 0,
        }
    always_include_core = bool(getattr(model, "always_include_core", True))
    plan = build_module_routing_plan(
        active_modules or module_names,
        module_names,
        always_include_core=always_include_core,
    )
    active_modules_set = set(plan.active_by_module)
    module_totals: dict[str, dict[str, int]] = {}
    shared_total = 0
    shared_trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        group = infer_parameter_group(name)
        if group.startswith("module:"):
            module_name = group.split(":", 1)[1]
            entry = module_totals.setdefault(module_name, {"total": 0, "trainable": 0})
            entry["total"] += count
            if parameter.requires_grad:
                entry["trainable"] += count
        else:
            shared_total += count
            if parameter.requires_grad:
                shared_trainable += count
    active_module_total = sum(
        values["total"]
        for module, values in module_totals.items()
        if module in active_modules_set
    )
    active_module_trainable = sum(
        values["trainable"]
        for module, values in module_totals.items()
        if module in active_modules_set
    )
    module_param_total = sum(values["total"] for values in module_totals.values())
    module_trainable_total = sum(values["trainable"] for values in module_totals.values())
    active = int(shared_total + active_module_total)
    active_trainable = int(shared_trainable + active_module_trainable)
    mop_block_type = getattr(model, "mop_block_type", "post_core_mlp")
    expert_count = int(getattr(model, "expert_count", len(module_names)) or len(module_names))
    active_experts = int(getattr(model, "active_experts", 1) or 1)
    active_expert_count = min(expert_count, max(1, len(active_modules_set), active_experts))
    shared_compute_ratio = float(getattr(model, "shared_depth_ratio", 1.0) or 1.0)
    if mop_block_type == "routed_ffn":
        expert_fraction = max(0.0, 1.0 - shared_compute_ratio)
        expert_compute_ratio = expert_fraction * (active_expert_count / max(1, expert_count))
        estimated_active_flop_ratio = min(1.0, shared_compute_ratio + expert_compute_ratio)
    else:
        expert_compute_ratio = None
        estimated_active_flop_ratio = active / max(1, totals["total"])
    estimated_backward_flop_ratio = (
        active_trainable / max(1, totals["total"])
        if totals["trainable"] < totals["total"]
        else estimated_active_flop_ratio
    )
    return {
        "active_params": active,
        "total_params": totals["total"],
        "active_ratio": active / max(1, totals["total"]),
        "active_trainable_params": active_trainable,
        "active_trainable_ratio": active_trainable / max(1, totals["total"]),
        "shared_params": shared_total,
        "shared_trainable_params": shared_trainable,
        "shared_frozen_params": max(0, shared_total - shared_trainable),
        "routed_module_params": active_module_total,
        "routed_module_trainable_params": active_module_trainable,
        "total_module_params": module_param_total,
        "total_module_trainable_params": module_trainable_total,
        "active_module_count": len(active_modules_set),
        "known_module_count": len(module_names),
        "routing_density": plan.density,
        "mop_block_type": mop_block_type,
        "active_expert_count": active_expert_count if mop_block_type == "routed_ffn" else None,
        "expert_count": expert_count if mop_block_type == "routed_ffn" else None,
        "expert_compute_ratio": expert_compute_ratio,
        "shared_compute_ratio": shared_compute_ratio if mop_block_type == "routed_ffn" else None,
        "estimated_active_flop_ratio": estimated_active_flop_ratio,
        "estimated_backward_flop_ratio": estimated_backward_flop_ratio,
    }


def fast_parameter_metadata(batch: dict[str, Any], model=None) -> dict[str, Any]:
    target_modules = batch.get("target_modules") if isinstance(batch, dict) else None
    batch_size = len(target_modules) if isinstance(target_modules, list) else 0
    active_adapters = [adapter_names_from_target_modules(modules) for modules in (target_modules or [])]
    active_conditions = [condition_names_from_target_modules(modules) for modules in (target_modules or [])]
    adapter_density = _density(active_adapters, getattr(model, "fast_adapter_names", ["default"]))
    condition_density = _density(active_conditions, getattr(model, "generated_condition_names", ["default"]))
    return {
        "batch_size": batch_size,
        "active_adapters": active_adapters,
        "active_conditions": active_conditions,
        "active_adapter_density": adapter_density,
        "generated_condition_density": condition_density,
    }


def _density(rows: list[list[str]], known: list[str]) -> float:
    if not rows:
        return 0.0
    return sum(len(set(row)) for row in rows) / max(1, len(rows) * len(known or ["default"]))
