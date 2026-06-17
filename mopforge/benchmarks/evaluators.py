"""Tiny CPU benchmark evaluators."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mopforge.benchmarks.config import BenchmarkConfig
from mopforge.benchmarks.metrics import count_by_key, safe_mean, safe_rate
from mopforge.data import CausalLMCollator, LessonCausalLMDataset, RouterCollator, RouterDataset
from mopforge.eval import evaluate_generated_code_for_lesson
from mopforge.kts import LessonStore
from mopforge.models import (
    TinyCausalTransformer,
    TinyMoPCausalTransformer,
    TinyModuleRouter,
    adapter_names_from_target_modules,
    condition_names_from_target_modules,
    predict_modules,
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
from mopforge.tokenization import (
    build_tokenizer,
    get_tokenizer_pad_token_id,
    get_tokenizer_vocab_size,
    tokenizer_spec_from_config,
)
from mopforge.training import (
    DEFAULT_KNOWN_MODULES,
    TrainableParameterPolicy,
    apply_trainable_policy,
    count_parameters,
    normalize_target_modules,
)


def evaluate_loss(config: BenchmarkConfig) -> dict[str, Any]:
    """Evaluate tiny causal-LM loss over KTS lessons."""

    torch = _require_torch()
    if CausalLMCollator is None:
        raise RuntimeError("PyTorch is required for CausalLMCollator.")
    runtime = _runtime(config)
    tokenizer = _build_tokenizer(config)
    lessons = _load_lessons(config, limit=config.max_examples)
    model = move_model_to_runtime(_build_model(config, tokenizer), runtime)
    checkpoint = _maybe_load_checkpoint(model, config.checkpoint_path)
    model.eval()

    from torch.utils.data import DataLoader

    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )
    losses = []
    with torch.no_grad():
        for batch in loader:
            kwargs = _active_kwargs(config, batch.get("target_modules"))
            batch = _move_lm_batch(batch, runtime.device_info.selected)
            _drop_lm_metadata(batch)
            with autocast_context(runtime):
                outputs = model(**batch, **kwargs)
            losses.append(_loss_value(outputs["loss"]))
    mean_loss = safe_mean(losses)
    return {
        "benchmark_type": "loss",
        "eval_loss_mean": mean_loss,
        "eval_loss_count": len(losses),
        "finite": bool(mean_loss is not None and math.isfinite(mean_loss)),
        "model_type": config.model_type,
        "examples": len(lessons),
        "checkpoint_loaded": checkpoint["loaded"],
        "checkpoint_error": checkpoint.get("error"),
        "source_run_id": config.run_id,
        "checkpoint_path": config.checkpoint_path,
        "runtime": runtime_metadata(runtime),
    }


def evaluate_code_correctness(config: BenchmarkConfig) -> dict[str, Any]:
    """Generate code for tiny KTS lessons and verify with lesson tests."""

    _require_torch()
    runtime = _runtime(config)
    tokenizer = _build_tokenizer(config)
    lessons = _load_lessons(
        config,
        limit=min(config.generation_examples, config.max_examples),
    )
    model = move_model_to_runtime(_build_model(config, tokenizer), runtime)
    checkpoint = _maybe_load_checkpoint(model, config.checkpoint_path)
    model.eval()

    results = []
    for lesson in lessons:
        active_modules = (
            list(lesson.target_modules)
            if config.model_type == "mop_oracle"
            else None
        )
        active_adapters = (
            adapter_names_from_target_modules(list(lesson.target_modules))
            if config.use_fast_adapters
            else None
        )
        active_conditions = (
            condition_names_from_target_modules(list(lesson.target_modules))
            if config.use_generated_params
            else None
        )
        result = evaluate_generated_code_for_lesson(
            model,
            tokenizer,
            lesson,
            max_new_tokens=config.generation_max_new_tokens,
            active_modules=active_modules,
            active_adapters=active_adapters,
            active_conditions=active_conditions,
            device=runtime.device_info.selected,
        )
        results.append(
            {
                "lesson_id": result["lesson_id"],
                "passed": bool(result["passed"]),
                "failure_type": result.get("failure_type"),
                "generated_preview": str(result.get("generated_text", ""))[:240],
                "candidate_preview": str(result.get("candidate_code", ""))[:240],
                "target_modules": list(result.get("target_modules", [])),
            }
        )
    pass_count = sum(1 for result in results if result["passed"])
    fail_count = len(results) - pass_count
    failures_by_type = count_by_key(
        [result for result in results if not result["passed"]],
        "failure_type",
    )
    return {
        "benchmark_type": "code_correctness",
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": safe_rate(pass_count, len(results)),
        "failures_by_type": failures_by_type,
        "examples": results,
        "checkpoint_loaded": checkpoint["loaded"],
        "checkpoint_error": checkpoint.get("error"),
        "source_run_id": config.run_id,
        "checkpoint_path": config.checkpoint_path,
        "runtime": runtime_metadata(runtime),
    }


def evaluate_router(config: BenchmarkConfig) -> dict[str, Any]:
    """Evaluate tiny learned-router predictions over KTS lessons."""

    torch = _require_torch()
    if RouterCollator is None:
        raise RuntimeError("PyTorch is required for RouterCollator.")
    runtime = _runtime(config)
    tokenizer = _build_tokenizer(config)
    lessons = _load_lessons(config, limit=config.max_examples)
    router = TinyModuleRouter(
        vocab_size=get_tokenizer_vocab_size(tokenizer),
        d_model=int(config.metadata.get("d_model", 16)),
        hidden_dim=int(config.metadata.get("router_hidden_dim", 32)),
        known_modules=DEFAULT_KNOWN_MODULES,
        pad_token_id=get_tokenizer_pad_token_id(tokenizer),
    )
    router = move_model_to_runtime(router, runtime)
    checkpoint = _maybe_load_checkpoint(router, config.checkpoint_path)
    router.eval()

    from torch.utils.data import DataLoader

    dataset = RouterDataset(
        lessons,
        tokenizer,
        known_modules=list(DEFAULT_KNOWN_MODULES),
        max_length=config.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=RouterCollator(tokenizer),
    )
    examples = []
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, runtime.device_info.selected)
            with autocast_context(runtime):
                outputs = router(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    module_mask=batch["module_mask"],
                )
            predicted = predict_modules(outputs["logits"], DEFAULT_KNOWN_MODULES)
            if predicted and all(isinstance(name, str) for name in predicted):
                predicted = [predicted]
            for lesson_id, target_modules, predicted_modules in zip(
                batch["lesson_id"],
                batch["target_modules"],
                predicted,
            ):
                target = normalize_target_modules(target_modules, DEFAULT_KNOWN_MODULES)
                prediction = normalize_target_modules(predicted_modules, DEFAULT_KNOWN_MODULES)
                _update_counts(target, prediction, tp, fp, fn)
                examples.append(
                    {
                        "lesson_id": lesson_id,
                        "target_modules": target,
                        "predicted_modules": prediction,
                        "exact_match": target == prediction,
                    }
                )
    exact = sum(1 for item in examples if item["exact_match"])
    return {
        "benchmark_type": "router",
        "exact_match_count": exact,
        "exact_match_rate": safe_rate(exact, len(examples)),
        "avg_predicted_modules": safe_mean(
            [len(item["predicted_modules"]) for item in examples]
        ),
        "avg_target_modules": safe_mean(
            [len(item["target_modules"]) for item in examples]
        ),
        "per_module_tp": dict(sorted(tp.items())),
        "per_module_fp": dict(sorted(fp.items())),
        "per_module_fn": dict(sorted(fn.items())),
        "examples": examples,
        "checkpoint_loaded": checkpoint["loaded"],
        "checkpoint_error": checkpoint.get("error"),
        "untrained_smoke": not checkpoint["loaded"],
        "source_run_id": config.run_id,
        "checkpoint_path": config.checkpoint_path,
        "runtime": runtime_metadata(runtime),
    }


def evaluate_parameter_efficiency(config: BenchmarkConfig) -> dict[str, Any]:
    """Report total/trainable/frozen parameters and group summaries."""

    _require_torch()
    runtime = _runtime(config)
    tokenizer = _build_tokenizer(config)
    model = _build_model(config, tokenizer)
    policy_mode = str(config.metadata.get("trainable_policy_mode", "all"))
    policy = TrainableParameterPolicy(
        mode=policy_mode,
        target_modules=config.target_modules,
        train_fast_adapters=config.use_fast_adapters,
        train_generated_params=config.use_generated_params,
        metadata={"source": "BenchmarkConfig"},
    )
    summaries = apply_trainable_policy(model, policy)
    counts = count_parameters(model)
    return {
        "benchmark_type": "parameter_efficiency",
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "frozen_params": counts["frozen"],
        "trainable_ratio": safe_rate(counts["trainable"], counts["total"]),
        "parameter_groups": [summary.to_dict() for summary in summaries],
        "model_type": config.model_type,
        "use_fast_adapters": bool(config.use_fast_adapters),
        "use_generated_params": bool(config.use_generated_params),
        "trainable_policy_mode": policy_mode,
        "source_run_id": config.run_id,
        "checkpoint_path": config.checkpoint_path,
        "runtime": runtime_metadata(runtime),
    }


def evaluate_composite(config: BenchmarkConfig) -> dict[str, Any]:
    """Run a tiny composite benchmark."""

    base = BenchmarkConfig.from_dict(config.to_dict())
    loss_config = BenchmarkConfig.from_dict({**base.to_dict(), "benchmark_type": "loss"})
    code_config = BenchmarkConfig.from_dict(
        {
            **base.to_dict(),
            "benchmark_type": "code_correctness",
            "generation_examples": min(base.generation_examples, 2),
            "generation_max_new_tokens": min(base.generation_max_new_tokens, 32),
        }
    )
    parameter_config = BenchmarkConfig.from_dict(
        {**base.to_dict(), "benchmark_type": "parameter_efficiency"}
    )
    runtime = _runtime(config)
    return {
        "benchmark_type": "composite",
        "parameter_efficiency": evaluate_parameter_efficiency(parameter_config),
        "loss": evaluate_loss(loss_config),
        "code_correctness": evaluate_code_correctness(code_config),
        "source_run_id": config.run_id,
        "checkpoint_path": config.checkpoint_path,
        "runtime": runtime_metadata(runtime),
    }


def _load_lessons(config: BenchmarkConfig, *, limit: int):
    path = Path(config.lesson_path)
    if not path.exists():
        raise FileNotFoundError(f"lesson_path does not exist: {path}")
    lessons = LessonStore(path).load_all()
    if not lessons:
        raise ValueError("Benchmark lesson store is empty.")
    return lessons[:limit]


def _build_tokenizer(config: BenchmarkConfig):
    spec = tokenizer_spec_from_config(config)
    return build_tokenizer(spec)


def _build_model(config: BenchmarkConfig, tokenizer):
    values = _model_values(config)
    if config.model_type == "dense":
        if TinyCausalTransformer is None:
            raise RuntimeError("PyTorch is required for TinyCausalTransformer.")
        return TinyCausalTransformer(
            vocab_size=get_tokenizer_vocab_size(tokenizer),
            use_fast_adapters=config.use_fast_adapters,
            fast_adapter_names=_adapter_names(config),
            use_generated_params=config.use_generated_params,
            generated_condition_names=_condition_names(config),
            **values,
        )
    if TinyMoPCausalTransformer is None:
        raise RuntimeError("PyTorch is required for TinyMoPCausalTransformer.")
    return TinyMoPCausalTransformer(
        vocab_size=get_tokenizer_vocab_size(tokenizer),
        module_names=list(DEFAULT_KNOWN_MODULES),
        use_fast_adapters=config.use_fast_adapters,
        fast_adapter_names=_adapter_names(config),
        use_generated_params=config.use_generated_params,
        generated_condition_names=_condition_names(config),
        **values,
    )


def _model_values(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "d_model": int(config.metadata.get("d_model", 16)),
        "n_heads": int(config.metadata.get("n_heads", 2)),
        "n_layers": int(config.metadata.get("n_layers", 1)),
        "max_seq_len": config.max_seq_len,
    }


def _adapter_names(config: BenchmarkConfig) -> list[str]:
    if config.target_modules:
        names = adapter_names_from_target_modules(config.target_modules)
        return names or ["default"]
    return ["coding", "debugging", "repair"] if config.use_fast_adapters else ["default"]


def _condition_names(config: BenchmarkConfig) -> list[str]:
    if config.generated_condition_names:
        return list(config.generated_condition_names)
    if config.target_modules:
        names = condition_names_from_target_modules(config.target_modules)
        return names or ["default"]
    return ["coding", "debugging", "repair"] if config.use_generated_params else ["default"]


def _active_kwargs(config: BenchmarkConfig, target_modules) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.model_type == "mop_oracle":
        kwargs["active_modules"] = target_modules
    if config.use_fast_adapters:
        kwargs["active_adapters"] = [
            adapter_names_from_target_modules(list(modules or []))
            for modules in list(target_modules or [])
        ]
    if config.use_generated_params:
        kwargs["active_conditions"] = [
            condition_names_from_target_modules(list(modules or []))
            for modules in list(target_modules or [])
        ]
    return kwargs


def _maybe_load_checkpoint(model, checkpoint_path: str | None) -> dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False}
    torch = _require_torch()
    try:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            state = (
                payload.get("model_state_dict")
                or payload.get("state_dict")
                or payload
            )
        else:
            state = payload
        model.load_state_dict(state)
    except Exception as exc:
        return {"loaded": False, "error": str(exc)}
    return {"loaded": True}


def _runtime(config: BenchmarkConfig):
    runtime = build_runtime_context(
        RuntimeConfig(
            device=config.device,
            precision=config.precision,
            enable_amp=config.enable_amp,
            allow_tf32=config.allow_tf32,
            deterministic=config.deterministic,
            compile_model=config.compile_model,
            require_device_available=config.require_device_available,
        )
    )
    apply_runtime_determinism(runtime, config.seed)
    return runtime


def _move_lm_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return move_batch_to_device(dict(batch), device)


def _drop_lm_metadata(batch: dict[str, Any]) -> None:
    for key in ("lesson_id", "target_modules", "metadata", "domain", "skill"):
        batch.pop(key, None)


def _loss_value(loss: Any) -> float:
    if loss is None:
        return float("nan")
    return float(loss.detach().cpu().item())


def _update_counts(target, prediction, tp, fp, fn) -> None:
    target_set = set(target)
    prediction_set = set(prediction)
    for module in target_set & prediction_set:
        tp[module] = tp.get(module, 0) + 1
    for module in prediction_set - target_set:
        fp[module] = fp.get(module, 0) + 1
    for module in target_set - prediction_set:
        fn[module] = fn.get(module, 0) + 1


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for benchmark evaluators.") from exc
    return torch
