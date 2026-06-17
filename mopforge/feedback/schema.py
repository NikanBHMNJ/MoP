"""Schemas for per-lesson feedback events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class LessonFeedbackRecord:
    """One observed model outcome for one lesson."""

    lesson_id: str
    run_id: str | None = None
    model_type: str | None = None
    curriculum_strategy: str | None = None
    passed: bool | None = None
    failure_type: str | None = None
    loss: float | None = None
    generated: bool = False
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate record fields and fill default timestamp."""

        self.validate()
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def validate(self) -> None:
        """Raise ``ValueError`` if this feedback record is malformed."""

        if not isinstance(self.lesson_id, str) or not self.lesson_id.strip():
            raise ValueError("lesson_id must be a non-empty string.")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise ValueError("passed must be a bool or None.")
        if self.loss is not None and not isinstance(self.loss, (int, float)):
            raise ValueError("loss must be numeric or None.")
        if not isinstance(self.generated, bool):
            raise ValueError("generated must be a bool.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable feedback dictionary."""

        return {
            "lesson_id": self.lesson_id,
            "run_id": self.run_id,
            "model_type": self.model_type,
            "curriculum_strategy": self.curriculum_strategy,
            "passed": self.passed,
            "failure_type": self.failure_type,
            "loss": self.loss,
            "generated": self.generated,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LessonFeedbackRecord":
        """Create a feedback record from a dictionary."""

        return cls(
            lesson_id=str(data["lesson_id"]),
            run_id=data.get("run_id"),
            model_type=data.get("model_type"),
            curriculum_strategy=data.get("curriculum_strategy"),
            passed=data.get("passed"),
            failure_type=data.get("failure_type"),
            loss=data.get("loss"),
            generated=bool(data.get("generated", False)),
            timestamp=data.get("timestamp"),
            metadata=dict(data.get("metadata", {})),
        )
