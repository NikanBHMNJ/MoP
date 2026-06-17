"""Schemas for module-specific training queue items."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ALLOWED_QUEUE_STATUSES = {"pending", "running", "done", "failed", "skipped"}


@dataclass(slots=True)
class TrainingQueueItem:
    """One local module-targeted training queue item."""

    item_id: str
    module: str
    lesson_id: str
    priority: float = 0.0
    status: str = "pending"
    source: str = "curriculum"
    run_id: str | None = None
    attempts: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill timestamps and validate the item."""

        if self.created_at is None:
            self.created_at = _now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if the item is malformed."""

        _require_non_empty(self.item_id, "item_id")
        _require_non_empty(self.module, "module")
        _require_non_empty(self.lesson_id, "lesson_id")
        if self.status not in ALLOWED_QUEUE_STATUSES:
            valid = ", ".join(sorted(ALLOWED_QUEUE_STATUSES))
            raise ValueError(f"status must be one of: {valid}.")
        if not isinstance(self.priority, (int, float)):
            raise ValueError("priority must be numeric.")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer.")
        if self.created_at is None or not isinstance(self.created_at, str):
            raise ValueError("created_at must be a timestamp string.")
        if self.updated_at is None or not isinstance(self.updated_at, str):
            raise ValueError("updated_at must be a timestamp string.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable queue item dictionary."""

        return {
            "item_id": self.item_id,
            "module": self.module,
            "lesson_id": self.lesson_id,
            "priority": float(self.priority),
            "status": self.status,
            "source": self.source,
            "run_id": self.run_id,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingQueueItem":
        """Create a queue item from a dictionary."""

        return cls(
            item_id=str(data["item_id"]),
            module=str(data["module"]),
            lesson_id=str(data["lesson_id"]),
            priority=float(data.get("priority", 0.0)),
            status=str(data.get("status", "pending")),
            source=str(data.get("source", "curriculum")),
            run_id=data.get("run_id"),
            attempts=int(data.get("attempts", 0)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=dict(data.get("metadata", {})),
        )


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
