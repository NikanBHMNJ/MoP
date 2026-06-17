"""GPU-aware MoP routing and fast-parameter metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mopforge.models.fast_adapters import adapter_names_from_target_modules
from mopforge.models.generated_params import condition_names_from_target_modules
from mopforge.training.parameter_policy import count_parameters
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


def build_module_routing_plan(active_modules, known_modules) -> ModuleRoutingPlan:
    known = list(known_modules)
    if active_modules is None:
        per_sample = [normalize_target_modules([], known)]
    elif isinstance(active_modules, str) or all(isinstance(item, str) for item in list(active_modules)):
        per_sample = [normalize_target_modules(active_modules, known)]
    else:
        per_sample = [normalize_target_modules(modules, known) for modules in active_modules]
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
        return {"active_params": totals["trainable"], "total_params": totals["total"], "active_ratio": 1.0}
    plan = build_module_routing_plan(active_modules or module_names, module_names)
    active_module_count = len(plan.active_by_module) or 1
    module_param_total = 0
    for name, parameter in model.named_parameters():
        if ".module_bank.blocks." in name:
            module_param_total += int(parameter.numel())
    shared = max(0, totals["total"] - module_param_total)
    per_module = module_param_total / max(1, len(module_names))
    active = int(shared + per_module * active_module_count)
    return {
        "active_params": active,
        "total_params": totals["total"],
        "active_ratio": active / max(1, totals["total"]),
        "routing_density": plan.density,
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
