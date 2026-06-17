"""Curriculum-driven tiny training runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.curriculum import CurriculumConfig, CurriculumScheduler
from mopforge.eval import evaluate_generated_code_for_lesson, summarize_generation_results
from mopforge.experiments import (
    TinyExperimentConfig,
    eval_tiny_dense,
    eval_tiny_mop_learned_router,
    eval_tiny_mop_oracle,
    train_tiny_dense,
    train_tiny_mop_learned_router,
    train_tiny_mop_oracle,
    train_tiny_router,
)
from mopforge.experiments.utils import split_lessons
from mopforge.kts import IndexedLessonStore, KnowledgeLesson, LessonStore
from mopforge.repair import build_repair_lessons_from_generation_results, write_repair_lessons
from mopforge.runs import RunRegistry, TinyTrainingRunConfig, TrainingRunRecord
from mopforge.tokenization import ByteTokenizer


@dataclass(slots=True)
class _TinyTrainingState:
    """Internal state from a curriculum training pass."""

    record: TrainingRunRecord
    model: Any
    router: Any | None
    tokenizer: ByteTokenizer
    plan: Any
    train_lessons: list[KnowledgeLesson]
    eval_lessons: list[KnowledgeLesson]


def run_tiny_training_from_curriculum(
    config: TinyTrainingRunConfig,
    registry: RunRegistry | None = None,
) -> TrainingRunRecord:
    """Run a tiny CPU-safe training/eval pass from a curriculum plan."""

    return _run_tiny_training_state(config, registry=registry).record


def _run_tiny_training_state(
    config: TinyTrainingRunConfig,
    registry: RunRegistry | None = None,
) -> _TinyTrainingState:
    """Run tiny training and keep model state for internal loop integrations."""

    if config.model_type not in {"dense", "mop_oracle", "mop_learned_router"}:
        raise ValueError("model_type must be dense, mop_oracle, or mop_learned_router.")

    started_at = _now()
    run_id = _make_run_id(config.run_name)
    registry = registry or RunRegistry()
    run_dir = registry.create_run_dir(run_id)

    store = _ensure_indexed_store(config)
    scheduler = CurriculumScheduler(indexed_store=store)
    plan = scheduler.build_plan(
        CurriculumConfig(
            strategy=config.curriculum_strategy,
            batch_size=config.batch_size,
            feedback_store_path=config.feedback_store_path,
        )
    )
    plan_path = plan.save_json(run_dir / "curriculum_plan.json")
    lessons = scheduler.load_lessons(plan)
    train_lessons, eval_lessons = _split_for_run(lessons, config.seed)

    experiment_config = _to_experiment_config(config)
    tokenizer = ByteTokenizer()
    metrics: dict[str, Any]
    artifacts = {"curriculum_plan_json": str(plan_path)}

    if config.model_type == "dense":
        router = None
        model, train_loss = train_tiny_dense(train_lessons, experiment_config, tokenizer)
        eval_metrics = eval_tiny_dense(model, eval_lessons, experiment_config, tokenizer)
        metrics = _metrics_from_eval(config, plan, train_lessons, eval_lessons, train_loss, eval_metrics)
        if config.run_generation_eval:
            metrics.update(_generation_metrics(model, tokenizer, eval_lessons, config))
    elif config.model_type == "mop_oracle":
        router = None
        model, train_loss = train_tiny_mop_oracle(train_lessons, experiment_config, tokenizer)
        eval_metrics = eval_tiny_mop_oracle(model, eval_lessons, experiment_config, tokenizer)
        metrics = _metrics_from_eval(config, plan, train_lessons, eval_lessons, train_loss, eval_metrics)
        if config.run_generation_eval:
            metrics.update(
                _generation_metrics(
                    model,
                    tokenizer,
                    eval_lessons,
                    config,
                    active_modules_mode="oracle",
                )
            )
    else:
        router, _router_loss = train_tiny_router(train_lessons, experiment_config, tokenizer)
        model, router, train_loss = train_tiny_mop_learned_router(
            train_lessons,
            experiment_config,
            tokenizer,
            router=router,
        )
        eval_metrics = eval_tiny_mop_learned_router(
            model,
            router,
            eval_lessons,
            experiment_config,
            tokenizer,
        )
        metrics = _metrics_from_eval(config, plan, train_lessons, eval_lessons, train_loss, eval_metrics)
        metrics["router_train_loss_last"] = _router_loss
        if config.run_generation_eval:
            metrics.update(_generation_metrics(model, tokenizer, eval_lessons, config))

    record = TrainingRunRecord(
        run_id=run_id,
        run_name=config.run_name,
        model_type=config.model_type,
        curriculum_strategy=config.curriculum_strategy,
        started_at=started_at,
        finished_at=_now(),
        config=config.to_dict(),
        metrics=metrics,
        artifacts=artifacts,
    )
    registry.save(record)
    return _TinyTrainingState(
        record=record,
        model=model,
        router=router if config.model_type == "mop_learned_router" else None,
        tokenizer=tokenizer,
        plan=plan,
        train_lessons=train_lessons,
        eval_lessons=eval_lessons,
    )


def _ensure_indexed_store(config: TinyTrainingRunConfig) -> IndexedLessonStore:
    lesson_path = Path(config.lesson_path)
    index_path = Path(config.index_path)
    if not lesson_path.exists():
        _build_default_lesson_store(lesson_path)
    return IndexedLessonStore(lesson_path, index_path, auto_rebuild=True)


def _build_default_lesson_store(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    repair_path = Path("data/repair_lessons.jsonl")
    if repair_path.exists():
        lessons.extend(LessonStore(repair_path).load_all())
    LessonStore(path).add_many(lesson for lesson in lessons if lesson.is_verified or lesson.verification.get("status") == "verified_target")


def _split_for_run(lessons, seed: int):
    if len(lessons) < 2:
        return list(lessons), list(lessons)
    train_lessons, eval_lessons = split_lessons(lessons, seed=seed)
    if not eval_lessons:
        eval_lessons = train_lessons[:]
    return train_lessons, eval_lessons


def _to_experiment_config(config: TinyTrainingRunConfig) -> TinyExperimentConfig:
    return TinyExperimentConfig(
        seed=config.seed,
        lesson_path=config.lesson_path,
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        eval_batches=config.eval_batches,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        router_train_steps=config.train_steps,
        learning_rate=config.learning_rate,
        run_generation_eval=config.run_generation_eval,
        generation_eval_examples=config.generation_eval_examples,
        max_new_tokens=config.max_new_tokens,
    )


def _metrics_from_eval(
    config: TinyTrainingRunConfig,
    plan,
    train_lessons,
    eval_lessons,
    train_loss,
    eval_metrics: dict[str, Any],
) -> dict[str, Any]:
    finite = bool(eval_metrics.get("finite", False))
    return {
        "model_type": config.model_type,
        "curriculum_strategy": config.curriculum_strategy,
        "train_loss_last": float(train_loss),
        "eval_loss_mean": float(eval_metrics["eval_loss_mean"]),
        "finite": finite,
        "train_examples": len(train_lessons),
        "eval_examples": len(eval_lessons),
        "curriculum_total": plan.total,
        "counts_by_skill": dict(plan.counts_by_skill),
        "counts_by_verification_status": dict(plan.counts_by_verification_status),
        **{
            key: value
            for key, value in eval_metrics.items()
            if key.startswith("router_")
        },
    }


def _generation_metrics(
    model,
    tokenizer,
    eval_lessons,
    config: TinyTrainingRunConfig,
    *,
    active_modules_mode: str = "none",
) -> dict[str, Any]:
    results = []
    for lesson in eval_lessons[: config.generation_eval_examples]:
        active_modules = list(lesson.target_modules) if active_modules_mode == "oracle" else None
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


def _make_run_id(run_name: str) -> str:
    safe_name = "".join(ch.lower() if ch.isalnum() else "-" for ch in run_name).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name or 'run'}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
