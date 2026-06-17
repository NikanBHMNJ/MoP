"""Oracle routing helpers for module-targeted lessons."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_KNOWN_MODULES: list[str] = [
    "core",
    "coding",
    "debugging",
    "math",
    "planning",
    "router",
    "fast_adapter",
]


def normalize_target_modules(
    target_modules: Iterable[str] | None,
    known_modules: Iterable[str],
    *,
    always_include_core: bool = True,
    strict: bool = False,
) -> list[str]:
    """Normalize requested target modules into a stable known-module list.

    Unknown modules are ignored by default. With ``strict=True`` they raise a
    ``ValueError``. Duplicates are removed, and the returned order follows
    ``known_modules`` for deterministic masks and model routing.
    """

    known_list = _validate_module_names(known_modules, "known_modules")
    known_set = set(known_list)
    requested = list(target_modules or [])
    requested_set: set[str] = set()
    unknown: list[str] = []

    for module in requested:
        if not isinstance(module, str) or not module.strip():
            raise ValueError("target_modules must contain non-empty strings.")
        if module not in known_set:
            if module not in unknown:
                unknown.append(module)
            continue
        requested_set.add(module)

    if unknown and strict:
        raise ValueError(f"Unknown target module(s): {', '.join(unknown)}.")

    if always_include_core and "core" in known_set:
        requested_set.add("core")

    return [module for module in known_list if module in requested_set]


def module_mask_from_targets(
    target_modules: Iterable[str] | None,
    known_modules: Iterable[str],
    *,
    always_include_core: bool = True,
    strict: bool = False,
) -> list[int]:
    """Return a stable 0/1 module mask aligned to ``known_modules``."""

    known_list = _validate_module_names(known_modules, "known_modules")
    normalized = set(
        normalize_target_modules(
            target_modules,
            known_list,
            always_include_core=always_include_core,
            strict=strict,
        )
    )
    return [1 if module in normalized else 0 for module in known_list]


def route_batch_with_router(
    router,
    batch: dict,
    known_modules: Iterable[str],
    *,
    threshold: float = 0.5,
    always_include_core: bool = True,
) -> list[list[str]]:
    """Predict active modules for a collated router batch.

    This is a tiny CPU-smoke integration helper. It expects a PyTorch router and
    batch tensors, but keeps imports out of the module so ``mopforge`` remains
    importable without PyTorch.
    """

    from mopforge.models.tiny_router import predict_modules

    outputs = router(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    predictions = predict_modules(
        outputs["logits"],
        list(known_modules),
        threshold=threshold,
        always_include_core=always_include_core,
    )
    if predictions and isinstance(predictions[0], str):
        return [predictions]
    return predictions


def _validate_module_names(
    modules: Iterable[str], field_name: str
) -> list[str]:
    module_list = list(modules)
    if not module_list:
        raise ValueError(f"{field_name} must contain at least one module name.")
    if not all(isinstance(module, str) and module.strip() for module in module_list):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(module_list) != len(set(module_list)):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return module_list
