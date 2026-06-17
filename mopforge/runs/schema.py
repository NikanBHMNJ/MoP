"""Schemas for tiny curriculum-driven training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TinyTrainingRunConfig:
    """CPU-safe configuration for tiny curriculum-driven training."""

    run_name: str = "tiny_curriculum_run"
    seed: int = 123
    model_type: str = "dense"
    curriculum_strategy: str = "balanced"
    lesson_path: str = "data/indexed_lessons.jsonl"
    index_path: str = "data/kts_index.sqlite"
    feedback_store_path: str | None = None
    batch_size: int = 2
    train_steps: int = 3
    eval_batches: int = 2
    max_seq_len: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    learning_rate: float = 1e-3
    run_generation_eval: bool = False
    generation_eval_examples: int = 2
    max_new_tokens: int = 64

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return asdict(self)


@dataclass(slots=True)
class TrainingRunRecord:
    """Metadata and metrics for one tiny training run."""

    run_id: str
    run_name: str
    model_type: str
    curriculum_strategy: str
    started_at: str
    finished_at: str | None
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable run record."""

        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "model_type": self.model_type,
            "curriculum_strategy": self.curriculum_strategy,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingRunRecord":
        """Create a run record from a dictionary."""

        return cls(
            run_id=str(data["run_id"]),
            run_name=str(data["run_name"]),
            model_type=str(data["model_type"]),
            curriculum_strategy=str(data["curriculum_strategy"]),
            started_at=str(data["started_at"]),
            finished_at=data.get("finished_at"),
            config=dict(data.get("config", {})),
            metrics=dict(data.get("metrics", {})),
            artifacts=dict(data.get("artifacts", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        """Write this run record to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainingRunRecord":
        """Load a run record from JSON."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
