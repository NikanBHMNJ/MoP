"""CPU-safe dense-vs-MoP comparison harness."""

from __future__ import annotations

import csv
import json
import math
from itertools import cycle
from pathlib import Path
from typing import Any

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import (
    CausalLMCollator,
    LessonCausalLMDataset,
    RouterCollator,
    RouterDataset,
)
from mopforge.eval import (
    evaluate_generated_code_for_lesson,
    summarize_generation_results,
)
from mopforge.experiments.config import TinyExperimentConfig
from mopforge.experiments.utils import mean, set_seed, split_lessons
from mopforge.kts import KnowledgeLesson, LessonStore
from mopforge.models import TinyCausalTransformer, TinyMoPCausalTransformer, TinyModuleRouter
from mopforge.tokenization import ByteTokenizer
from mopforge.training import normalize_target_modules, route_batch_with_router


REQUIRED_RESULT_KEYS = {
    "model",
    "routing",
    "train_loss_last",
    "eval_loss_mean",
    "finite",
    "train_examples",
    "eval_examples",
}


def load_or_generate_lessons(path: str | Path) -> list[KnowledgeLesson]:
    """Load verified coding lessons, generating the demo set if needed."""

    lesson_path = Path(path)
    if not lesson_path.exists():
        lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
        LessonStore(lesson_path).add_many(
            lesson for lesson in lessons if lesson.is_verified
        )
    return LessonStore(lesson_path).load_all()


def train_tiny_dense(
    train_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
) -> tuple[Any, float]:
    """Train the dense baseline for a few CPU smoke steps."""

    torch = _require_torch()
    _require_model(TinyCausalTransformer, "TinyCausalTransformer")
    tokenizer = tokenizer or ByteTokenizer()
    set_seed(config.seed)
    model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = _lm_loader(train_lessons, tokenizer, config)
    last_loss = _train_lm_steps(
        model,
        optimizer,
        loader,
        config.train_steps,
        active_modules_fn=None,
    )
    return model, last_loss


def eval_tiny_dense(
    model: Any,
    eval_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
) -> dict[str, Any]:
    """Evaluate the dense baseline for a few CPU smoke batches."""

    tokenizer = tokenizer or ByteTokenizer()
    eval_loss = _eval_lm(
        model,
        _lm_loader(eval_lessons, tokenizer, config),
        config.eval_batches,
        active_modules_fn=None,
    )
    return {
        "model": "tiny_dense",
        "routing": "none",
        "eval_loss_mean": eval_loss,
        "eval_examples": len(eval_lessons),
        "finite": math.isfinite(eval_loss),
    }


def train_tiny_mop_oracle(
    train_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
) -> tuple[Any, float]:
    """Train TinyMoP using oracle lesson target modules."""

    torch = _require_torch()
    _require_model(TinyMoPCausalTransformer, "TinyMoPCausalTransformer")
    tokenizer = tokenizer or ByteTokenizer()
    set_seed(config.seed)
    model = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        module_names=config.known_modules,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = _lm_loader(train_lessons, tokenizer, config)
    last_loss = _train_lm_steps(
        model,
        optimizer,
        loader,
        config.train_steps,
        active_modules_fn=lambda batch: batch["target_modules"],
    )
    return model, last_loss


def eval_tiny_mop_oracle(
    model: Any,
    eval_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
) -> dict[str, Any]:
    """Evaluate TinyMoP using oracle lesson target modules."""

    tokenizer = tokenizer or ByteTokenizer()
    eval_loss = _eval_lm(
        model,
        _lm_loader(eval_lessons, tokenizer, config),
        config.eval_batches,
        active_modules_fn=lambda batch: batch["target_modules"],
    )
    return {
        "model": "tiny_mop",
        "routing": "oracle",
        "eval_loss_mean": eval_loss,
        "eval_examples": len(eval_lessons),
        "finite": math.isfinite(eval_loss),
    }


def train_tiny_router(
    train_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
) -> tuple[Any, float]:
    """Train the tiny learned router for a few CPU smoke steps."""

    torch = _require_torch()
    _require_model(TinyModuleRouter, "TinyModuleRouter")
    tokenizer = tokenizer or ByteTokenizer()
    set_seed(config.seed)
    router = TinyModuleRouter(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        hidden_dim=config.router_hidden_dim,
        known_modules=config.known_modules,
        pad_token_id=tokenizer.pad_token_id,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.learning_rate)
    loader = _router_loader(train_lessons, tokenizer, config)
    router.train()
    last_loss = float("nan")
    for step, batch in enumerate(cycle(loader), start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = router(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            module_mask=batch["module_mask"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        last_loss = _loss_value(loss)
        if step >= config.router_train_steps:
            break
    return router, last_loss


def train_tiny_mop_learned_router(
    train_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
    router: Any | None = None,
) -> tuple[Any, Any, float]:
    """Train TinyMoP using modules predicted by the learned router."""

    torch = _require_torch()
    _require_model(TinyMoPCausalTransformer, "TinyMoPCausalTransformer")
    tokenizer = tokenizer or ByteTokenizer()
    if router is None:
        router, _ = train_tiny_router(train_lessons, config, tokenizer)

    set_seed(config.seed)
    model = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        module_names=config.known_modules,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lm_loader = _lm_loader(train_lessons, tokenizer, config)
    router_loader = _router_loader(train_lessons, tokenizer, config)
    router.eval()
    model.train()
    last_loss = float("nan")
    for step, (lm_batch, router_batch) in enumerate(
        zip(cycle(lm_loader), cycle(router_loader)), start=1
    ):
        active_modules = route_batch_with_router(
            router, router_batch, config.known_modules
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            lm_batch["input_ids"],
            attention_mask=lm_batch["attention_mask"],
            labels=lm_batch["labels"],
            active_modules=active_modules,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        last_loss = _loss_value(loss)
        if step >= config.train_steps:
            break
    return model, router, last_loss


def eval_tiny_mop_learned_router(
    model: Any,
    router: Any,
    eval_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer | None = None,
) -> dict[str, Any]:
    """Evaluate TinyMoP using modules predicted by a learned router."""

    tokenizer = tokenizer or ByteTokenizer()
    lm_loader = _lm_loader(eval_lessons, tokenizer, config)
    router_loader = _router_loader(eval_lessons, tokenizer, config)
    losses: list[float] = []
    exact_matches = 0
    sample_count = 0
    predicted_counts: list[float] = []
    target_counts: list[float] = []
    router.eval()
    model.eval()
    torch = _require_torch()
    with torch.no_grad():
        for batch_index, (lm_batch, router_batch) in enumerate(
            zip(lm_loader, router_loader), start=1
        ):
            active_modules = route_batch_with_router(
                router, router_batch, config.known_modules
            )
            outputs = model(
                lm_batch["input_ids"],
                attention_mask=lm_batch["attention_mask"],
                labels=lm_batch["labels"],
                active_modules=active_modules,
            )
            losses.append(_loss_value(outputs["loss"]))

            for predicted, target in zip(
                active_modules, router_batch["target_modules"]
            ):
                normalized_target = normalize_target_modules(
                    target, config.known_modules
                )
                exact_matches += int(predicted == normalized_target)
                sample_count += 1
                predicted_counts.append(float(len(predicted)))
                target_counts.append(float(len(normalized_target)))

            if batch_index >= config.eval_batches:
                break

    eval_loss = mean(losses)
    return {
        "model": "tiny_mop",
        "routing": "learned_router",
        "eval_loss_mean": eval_loss,
        "eval_examples": len(eval_lessons),
        "finite": math.isfinite(eval_loss),
        "router_exact_match_count": exact_matches,
        "router_eval_examples": sample_count,
        "router_avg_predicted_modules": mean(predicted_counts),
        "router_avg_target_modules": mean(target_counts),
    }


def run_tiny_comparison(
    lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig | None = None,
) -> list[dict[str, Any]]:
    """Run dense, oracle-MoP, and learned-router-MoP comparisons."""

    config = config or TinyExperimentConfig()
    tokenizer = ByteTokenizer()
    train_lessons, eval_lessons = split_lessons(
        lessons, train_fraction=config.train_fraction, seed=config.seed
    )

    dense_model, dense_train_loss = train_tiny_dense(
        train_lessons, config, tokenizer
    )
    dense = eval_tiny_dense(dense_model, eval_lessons, config, tokenizer)
    dense["train_loss_last"] = dense_train_loss
    dense["train_examples"] = len(train_lessons)
    dense["finite"] = dense["finite"] and math.isfinite(dense_train_loss)
    if config.run_generation_eval:
        dense.update(
            _generation_metrics(
                dense_model,
                None,
                eval_lessons,
                config,
                tokenizer,
                active_modules_mode="none",
            )
        )

    oracle_model, oracle_train_loss = train_tiny_mop_oracle(
        train_lessons, config, tokenizer
    )
    oracle = eval_tiny_mop_oracle(oracle_model, eval_lessons, config, tokenizer)
    oracle["train_loss_last"] = oracle_train_loss
    oracle["train_examples"] = len(train_lessons)
    oracle["finite"] = oracle["finite"] and math.isfinite(oracle_train_loss)
    if config.run_generation_eval:
        oracle.update(
            _generation_metrics(
                oracle_model,
                None,
                eval_lessons,
                config,
                tokenizer,
                active_modules_mode="oracle",
            )
        )

    router, router_train_loss = train_tiny_router(train_lessons, config, tokenizer)
    learned_model, router, learned_train_loss = train_tiny_mop_learned_router(
        train_lessons, config, tokenizer, router=router
    )
    learned = eval_tiny_mop_learned_router(
        learned_model, router, eval_lessons, config, tokenizer
    )
    learned["train_loss_last"] = learned_train_loss
    learned["train_examples"] = len(train_lessons)
    learned["router_train_loss_last"] = router_train_loss
    learned["finite"] = (
        learned["finite"]
        and math.isfinite(learned_train_loss)
        and math.isfinite(router_train_loss)
    )
    if config.run_generation_eval:
        learned.update(
            _generation_metrics(
                learned_model,
                router,
                eval_lessons,
                config,
                tokenizer,
                active_modules_mode="learned_router",
            )
        )

    return [_ordered_result(dense), _ordered_result(oracle), _ordered_result(learned)]


def write_results(
    results: list[dict[str, Any]],
    output_dir: str | Path = "outputs",
    *,
    write_csv: bool = True,
) -> tuple[Path, Path | None]:
    """Write comparison results to JSON and optionally CSV."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "tiny_comparison_results.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = None
    if write_csv:
        csv_path = path / "tiny_comparison_results.csv"
        keys = sorted({key for result in results for key in result})
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    return json_path, csv_path


def _lm_loader(
    lessons: list[KnowledgeLesson],
    tokenizer: ByteTokenizer,
    config: TinyExperimentConfig,
):
    _require_torch()
    if CausalLMCollator is None:
        raise RuntimeError("PyTorch is required for CausalLMCollator.")
    from torch.utils.data import DataLoader

    dataset = LessonCausalLMDataset(
        lessons, tokenizer, max_length=config.max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )


def _router_loader(
    lessons: list[KnowledgeLesson],
    tokenizer: ByteTokenizer,
    config: TinyExperimentConfig,
):
    _require_torch()
    if RouterCollator is None:
        raise RuntimeError("PyTorch is required for RouterCollator.")
    from torch.utils.data import DataLoader

    dataset = RouterDataset(
        lessons,
        tokenizer,
        known_modules=config.known_modules,
        max_length=config.max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=RouterCollator(tokenizer),
    )


def _train_lm_steps(
    model: Any,
    optimizer: Any,
    loader: Any,
    steps: int,
    active_modules_fn,
) -> float:
    model.train()
    last_loss = float("nan")
    for step, batch in enumerate(cycle(loader), start=1):
        optimizer.zero_grad(set_to_none=True)
        kwargs = {}
        if active_modules_fn is not None:
            kwargs["active_modules"] = active_modules_fn(batch)
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            **kwargs,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        last_loss = _loss_value(loss)
        if step >= steps:
            break
    return last_loss


def _eval_lm(
    model: Any,
    loader: Any,
    eval_batches: int,
    active_modules_fn,
) -> float:
    torch = _require_torch()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            kwargs = {}
            if active_modules_fn is not None:
                kwargs["active_modules"] = active_modules_fn(batch)
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                **kwargs,
            )
            losses.append(_loss_value(outputs["loss"]))
            if batch_index >= eval_batches:
                break
    return mean(losses)


def _generation_metrics(
    model: Any,
    router: Any | None,
    eval_lessons: list[KnowledgeLesson],
    config: TinyExperimentConfig,
    tokenizer: ByteTokenizer,
    *,
    active_modules_mode: str,
) -> dict[str, Any]:
    lessons = eval_lessons[: config.generation_eval_examples]
    results = []
    predicted_modules_by_lesson: list[list[str]] | None = None

    if active_modules_mode == "learned_router":
        if router is None:
            raise ValueError("router is required for learned-router generation eval.")
        predicted_modules_by_lesson = []
        for router_batch in _router_loader(lessons, tokenizer, config):
            predicted_modules_by_lesson.extend(
                route_batch_with_router(
                    router,
                    router_batch,
                    config.known_modules,
                )
            )
            if len(predicted_modules_by_lesson) >= len(lessons):
                break

    for index, lesson in enumerate(lessons):
        active_modules = None
        if active_modules_mode == "oracle":
            active_modules = list(lesson.target_modules)
        elif active_modules_mode == "learned_router" and predicted_modules_by_lesson:
            active_modules = predicted_modules_by_lesson[index]
        results.append(
            evaluate_generated_code_for_lesson(
                model,
                tokenizer,
                lesson,
                max_new_tokens=config.max_new_tokens,
                active_modules=active_modules,
            )
        )
    return summarize_generation_results(results)


def _loss_value(loss: Any) -> float:
    if loss is None:
        return float("nan")
    return float(loss.detach().cpu().item())


def _ordered_result(result: dict[str, Any]) -> dict[str, Any]:
    ordered = {
        "model": result["model"],
        "routing": result["routing"],
        "train_loss_last": float(result["train_loss_last"]),
        "eval_loss_mean": float(result["eval_loss_mean"]),
        "finite": bool(result["finite"]),
        "train_examples": int(result["train_examples"]),
        "eval_examples": int(result["eval_examples"]),
    }
    for key in sorted(set(result) - set(ordered)):
        ordered[key] = result[key]
    return ordered


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for tiny comparison experiments.") from exc
    return torch


def _require_model(model_cls: Any, name: str) -> None:
    if model_cls is None:
        raise RuntimeError(f"PyTorch is required for {name}.")
