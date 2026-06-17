"""Schemas for local artifact records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ALLOWED_ARTIFACT_KINDS = {
    "checkpoint",
    "metrics",
    "curriculum_plan",
    "generation_eval",
    "feedback_export",
    "queue_item",
    "config",
    "other",
}


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata for one local artifact file."""

    artifact_id: str
    kind: str
    path: str
    run_id: str | None = None
    queue_item_id: str | None = None
    model_type: str | None = None
    module: str | None = None
    step: int | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill default timestamp and validate the record."""

        if self.created_at is None:
            self.created_at = _now()
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if this artifact record is malformed."""

        _require_non_empty(self.artifact_id, "artifact_id")
        _require_non_empty(self.path, "path")
        if self.kind not in ALLOWED_ARTIFACT_KINDS:
            valid = ", ".join(sorted(ALLOWED_ARTIFACT_KINDS))
            raise ValueError(f"kind must be one of: {valid}.")
        if self.step is not None and (type(self.step) is not int or self.step < 0):
            raise ValueError("step must be a non-negative integer or None.")
        if self.created_at is None or not isinstance(self.created_at, str):
            raise ValueError("created_at must be a timestamp string.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact dictionary."""

        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "run_id": self.run_id,
            "queue_item_id": self.queue_item_id,
            "model_type": self.model_type,
            "module": self.module,
            "step": self.step,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        """Create an artifact record from a dictionary."""

        step = data.get("step")
        return cls(
            artifact_id=str(data["artifact_id"]),
            kind=str(data["kind"]),
            path=str(data["path"]),
            run_id=data.get("run_id"),
            queue_item_id=data.get("queue_item_id"),
            model_type=data.get("model_type"),
            module=data.get("module"),
            step=int(step) if step is not None else None,
            created_at=data.get("created_at"),
            metadata=dict(data.get("metadata", {})),
        )


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
