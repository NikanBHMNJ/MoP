"""Feedback-weighted tiny retraining loop."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.data import RouterCollator, RouterDataset
from mopforge.eval import evaluate_generated_code_for_lesson, write_generation_eval_results
from mopforge.experiments import TinyExperimentConfig
from mopforge.feedback import LessonFeedbackStore, feedback_records_from_generation_eval
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.runs import RunRegistry, TinyTrainingRunConfig
from mopforge.training import route_batch_with_router
from mopforge.training.runner import _run_tiny_training_state


@dataclass(slots=True)
class FeedbackRetrainingConfig:
    """CPU-safe settings for one tiny feedback-weighted retraining loop."""

    loop_name: str = "tiny_feedback_retraining"
    seed: int = 123
    model_type: str = "dense"
    lesson_path: str = "data/indexed_lessons.jsonl"
    index_path: str = "data/kts_index.sqlite"
    feedback_store_path: str = "data/lesson_feedback.sqlite"
    run_registry_root: str = "runs"
    curriculum_strategy: str = "feedback_weighted"
    batch_size: int = 2
    train_steps: int = 3
    eval_batches: int = 1
    generation_eval_examples: int = 2
    max_new_tokens: int = 64
    max_seq_len: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    learning_rate: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)


@dataclass(slots=True)
class FeedbackRetrainingResult:
    """Summary and artifacts for one feedback retraining loop run."""

    loop_id: str
    loop_name: str
    model_type: str
    curriculum_strategy: str
    train_run_id: str | None
    feedback_records_added: int
    eval_examples: int
    pass_count: int
    fail_count: int
    failures_by_type: dict[str, int]
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "loop_id": self.loop_id,
            "loop_name": self.loop_name,
            "model_type": self.model_type,
            "curriculum_strategy": self.curriculum_strategy,
            "train_run_id": self.train_run_id,
            "feedback_records_added": self.feedback_records_added,
            "eval_examples": self.eval_examples,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "failures_by_type": dict(self.failures_by_type),
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackRetrainingResult":
        """Create a loop result from a dictionary."""

        return cls(
            loop_id=str(data["loop_id"]),
            loop_name=str(data["loop_name"]),
            model_type=str(data["model_type"]),
            curriculum_strategy=str(data["curriculum_strategy"]),
            train_run_id=data.get("train_run_id"),
            feedback_records_added=int(data["feedback_records_added"]),
            eval_examples=int(data["eval_examples"]),
            pass_count=int(data["pass_count"]),
            fail_count=int(data["fail_count"]),
            failures_by_type=dict(data.get("failures_by_type", {})),
            artifacts=dict(data.get("artifacts", {})),
            metrics=dict(data.get("metrics", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        """Write this loop result to JSON and return the output path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> "FeedbackRetrainingResult":
        """Load a loop result from JSON."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run_feedback_retraining_loop(
    config: FeedbackRetrainingConfig,
) -> FeedbackRetrainingResult:
    """Run one CPU-safe feedback-weighted retraining loop."""

    lesson_path = Path(config.lesson_path)
    if not lesson_path.exists():
        raise FileNotFoundError(
            f"Lesson JSONL does not exist: {lesson_path}. "
            "Generate or index lessons before running the feedback loop."
        )

    indexed_store = IndexedLessonStore(
        lesson_path,
        config.index_path,
        auto_rebuild=True,
    )
    if indexed_store.count() == 0:
        raise ValueError(f"No lessons found in {lesson_path}.")

    loop_id = _make_loop_id(config.loop_name)
    loop_dir = Path(config.run_registry_root) / loop_id
    loop_dir.mkdir(parents=True, exist_ok=True)

    feedback_store = LessonFeedbackStore(config.feedback_store_path)
    before = _feedback_snapshot(feedback_store)

    registry = RunRegistry(config.run_registry_root)
    training_state = _run_tiny_training_state(
        _training_config_from_loop(config),
        registry=registry,
    )

    eval_lessons = _select_eval_lessons(
        training_state.eval_lessons,
        training_state.train_lessons,
        config.generation_eval_examples,
    )
    eval_results = _evaluate_after_retraining(
        training_state,
        eval_lessons,
        config,
    )
    generation_eval_path = write_generation_eval_results(
        eval_results,
        loop_dir / "generation_eval_after_retraining.json",
    )

    feedback_records = feedback_records_from_generation_eval(
        eval_results,
        run_id=training_state.record.run_id,
        model_type=config.model_type,
        curriculum_strategy=config.curriculum_strategy,
    )
    added = feedback_store.add_many(feedback_records)
    feedback_export_path = feedback_store.export_json(
        loop_dir / "feedback_export_after_retraining.json"
    )
    after = _feedback_snapshot(feedback_store)
    feedback_delta = summarize_feedback_delta(before, after)

    pass_count = sum(1 for result in eval_results if result.get("passed") is True)
    fail_count = len(eval_results) - pass_count
    failures_by_type = Counter(
        str(result.get("failure_type") or "unknown")
        for result in eval_results
        if result.get("passed") is not True
    )

    artifacts = {
        "generation_eval_after_retraining_json": str(generation_eval_path),
        "feedback_export_after_retraining_json": str(feedback_export_path),
    }
    if "run_json" in training_state.record.artifacts:
        artifacts["train_run_json"] = str(training_state.record.artifacts["run_json"])
    if "metrics_json" in training_state.record.artifacts:
        artifacts["train_metrics_json"] = str(
            training_state.record.artifacts["metrics_json"]
        )
    if "curriculum_plan_json" in training_state.record.artifacts:
        artifacts["train_curriculum_plan_json"] = str(
            training_state.record.artifacts["curriculum_plan_json"]
        )

    metrics = {
        "training_metrics": dict(training_state.record.metrics),
        "feedback_delta": feedback_delta,
        "curriculum_total": training_state.plan.total,
        "curriculum_first_ids": list(training_state.plan.lesson_ids[:5]),
    }
    result = FeedbackRetrainingResult(
        loop_id=loop_id,
        loop_name=config.loop_name,
        model_type=config.model_type,
        curriculum_strategy=config.curriculum_strategy,
        train_run_id=training_state.record.run_id,
        feedback_records_added=added,
        eval_examples=len(eval_results),
        pass_count=pass_count,
        fail_count=fail_count,
        failures_by_type=dict(sorted(failures_by_type.items())),
        artifacts=artifacts,
        metrics=metrics,
    )
    loop_result_path = loop_dir / "loop_result.json"
    result.artifacts["loop_result_json"] = str(loop_result_path)
    result.save_json(loop_result_path)
    return result


def summarize_feedback_delta(
    before_summary: dict[str, Any],
    after_summary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize feedback count and failure-count changes."""

    before_count = int(before_summary.get("feedback_count", 0) or 0)
    after_count = int(after_summary.get("feedback_count", 0) or 0)
    before_failures = dict(before_summary.get("failure_counts", {}) or {})
    after_failures = dict(after_summary.get("failure_counts", {}) or {})
    return {
        "feedback_count_before": before_count,
        "feedback_count_after": after_count,
        "new_feedback_records": after_count - before_count,
        "failure_counts_before": before_failures,
        "failure_counts_after": after_failures,
    }


def _training_config_from_loop(
    config: FeedbackRetrainingConfig,
) -> TinyTrainingRunConfig:
    return TinyTrainingRunConfig(
        run_name=f"{config.loop_name}_train",
        seed=config.seed,
        model_type=config.model_type,
        curriculum_strategy=config.curriculum_strategy,
        lesson_path=config.lesson_path,
        index_path=config.index_path,
        feedback_store_path=config.feedback_store_path,
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        eval_batches=config.eval_batches,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        learning_rate=config.learning_rate,
        run_generation_eval=False,
        generation_eval_examples=config.generation_eval_examples,
        max_new_tokens=config.max_new_tokens,
    )


def _select_eval_lessons(
    eval_lessons: list[KnowledgeLesson],
    train_lessons: list[KnowledgeLesson],
    count: int,
) -> list[KnowledgeLesson]:
    if count <= 0:
        return []
    source = eval_lessons or train_lessons
    return list(source[:count])


def _evaluate_after_retraining(
    training_state,
    eval_lessons: list[KnowledgeLesson],
    config: FeedbackRetrainingConfig,
) -> list[dict[str, Any]]:
    predicted_modules = None
    if config.model_type == "mop_learned_router" and training_state.router is not None:
        predicted_modules = _predict_modules_for_lessons(
            training_state.router,
            training_state.tokenizer,
            eval_lessons,
            config,
        )

    results: list[dict[str, Any]] = []
    for index, lesson in enumerate(eval_lessons):
        active_modules = None
        if config.model_type == "mop_oracle":
            active_modules = list(lesson.target_modules)
        elif predicted_modules is not None and index < len(predicted_modules):
            active_modules = predicted_modules[index]

        results.append(
            evaluate_generated_code_for_lesson(
                training_state.model,
                training_state.tokenizer,
                lesson,
                max_new_tokens=config.max_new_tokens,
                active_modules=active_modules,
            )
        )
    return results


def _predict_modules_for_lessons(
    router,
    tokenizer,
    lessons: list[KnowledgeLesson],
    config: FeedbackRetrainingConfig,
) -> list[list[str]]:
    if not lessons:
        return []
    if RouterDataset is None or RouterCollator is None:
        return []

    import torch
    from torch.utils.data import DataLoader

    experiment_config = TinyExperimentConfig(
        seed=config.seed,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    )
    dataset = RouterDataset(
        lessons,
        tokenizer,
        known_modules=experiment_config.known_modules,
        max_length=config.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=RouterCollator(tokenizer),
    )
    predictions: list[list[str]] = []
    router.eval()
    with torch.no_grad():
        for batch in loader:
            predictions.extend(
                route_batch_with_router(
                    router,
                    batch,
                    experiment_config.known_modules,
                )
            )
            if len(predictions) >= len(lessons):
                break
    return predictions[: len(lessons)]


def _feedback_snapshot(store: LessonFeedbackStore) -> dict[str, Any]:
    return {
        "feedback_count": store.count(),
        "failure_counts": store.failure_counts_by_type(),
    }


def _make_loop_id(loop_name: str) -> str:
    safe_name = "".join(ch.lower() if ch.isalnum() else "-" for ch in loop_name).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_name or 'loop'}-{uuid4().hex[:8]}"
