"""Full training checkpoint schema and torch save/load helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.lifecycle.rng import capture_rng_state


CHECKPOINT_FORMAT_VERSION = "1"


@dataclass(slots=True)
class TrainingCheckpointRecord:
    """Metadata for a full local training checkpoint."""

    checkpoint_id: str
    run_id: str
    step: int
    epoch: int | None = None
    model_type: str | None = None
    training_kind: str | None = None
    path: str = ""
    created_at: str = ""
    artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now()
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if this record is malformed."""

        _require_non_empty(self.checkpoint_id, "checkpoint_id")
        _require_non_empty(self.run_id, "run_id")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer.")
        if self.epoch is not None and (type(self.epoch) is not int or self.epoch < 0):
            raise ValueError("epoch must be a non-negative integer or None.")
        if self.path is not None and not isinstance(self.path, str):
            raise ValueError("path must be a string.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a timestamp string.")
        if self.metadata is None or not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record dictionary."""

        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "step": self.step,
            "epoch": self.epoch,
            "model_type": self.model_type,
            "training_kind": self.training_kind,
            "path": self.path,
            "created_at": self.created_at,
            "artifact_id": self.artifact_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingCheckpointRecord":
        """Create a record from a dictionary."""

        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            run_id=str(data["run_id"]),
            step=int(data["step"]),
            epoch=int(data["epoch"]) if data.get("epoch") is not None else None,
            model_type=data.get("model_type"),
            training_kind=data.get("training_kind"),
            path=str(data.get("path", "")),
            created_at=str(data.get("created_at", "")),
            artifact_id=data.get("artifact_id"),
            metadata=dict(data.get("metadata", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        """Write this record to JSON and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainingCheckpointRecord":
        """Load a checkpoint record JSON file."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_full_training_checkpoint(
    *,
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    trainer_state=None,
    config=None,
    tokenizer_spec=None,
    parameter_policy=None,
    adapter_metadata=None,
    generated_metadata=None,
    rng_state=None,
    metadata=None,
    run_id: str | None = None,
    checkpoint_id: str | None = None,
    global_step: int | None = None,
    epoch: int | None = None,
    model_type: str | None = None,
    training_kind: str | None = None,
    artifact_id: str | None = None,
) -> TrainingCheckpointRecord:
    """Save a full local training checkpoint and return its typed record."""

    torch = _require_torch()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dict = dict(metadata or {})
    trainer_state_dict = _to_plain_dict(trainer_state)
    config_dict = _to_plain_dict(config)
    tokenizer_dict = _to_plain_dict(tokenizer_spec)
    parameter_policy_dict = _to_plain_dict(parameter_policy)
    adapter_metadata_dict = _to_plain_dict(adapter_metadata)
    generated_metadata_dict = _to_plain_dict(generated_metadata)

    resolved_step = _resolve_step(global_step, trainer_state_dict, metadata_dict)
    resolved_epoch = _resolve_epoch(epoch, trainer_state_dict, metadata_dict)
    resolved_run_id = (
        run_id
        or metadata_dict.get("run_id")
        or trainer_state_dict.get("run_id")
        or config_dict.get("run_id")
        or "unknown-run"
    )
    resolved_model_type = (
        model_type or metadata_dict.get("model_type") or config_dict.get("model_type")
    )
    resolved_training_kind = (
        training_kind
        or metadata_dict.get("training_kind")
        or config_dict.get("training_kind")
    )
    resolved_checkpoint_id = (
        checkpoint_id
        or metadata_dict.get("checkpoint_id")
        or artifact_id
        or f"full-checkpoint-{uuid4().hex[:12]}"
    )
    resolved_rng_state = capture_rng_state() if rng_state is None else rng_state
    created_at = _now()

    record = TrainingCheckpointRecord(
        checkpoint_id=str(resolved_checkpoint_id),
        run_id=str(resolved_run_id),
        step=resolved_step,
        epoch=resolved_epoch,
        model_type=resolved_model_type,
        training_kind=resolved_training_kind,
        path=str(checkpoint_path),
        created_at=created_at,
        artifact_id=artifact_id,
        metadata=dict(metadata_dict),
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_id": record.checkpoint_id,
        "run_id": record.run_id,
        "global_step": record.step,
        "epoch": record.epoch,
        "model_type": record.model_type,
        "training_kind": record.training_kind,
        "created_at": record.created_at,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "trainer_state": trainer_state_dict,
        "config": config_dict,
        "tokenizer_spec": tokenizer_dict,
        "parameter_policy": parameter_policy_dict,
        "adapter_metadata": adapter_metadata_dict,
        "generated_metadata": generated_metadata_dict,
        "rng_state": resolved_rng_state,
        "metadata": metadata_dict,
        "record": record.to_dict(),
    }
    torch.save(payload, checkpoint_path)
    return record


def load_full_training_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load and validate a full training checkpoint payload."""

    torch = _require_torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Full training checkpoint payload must be a dictionary.")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format_version: "
            f"{payload.get('format_version')!r}."
        )
    if "model_state_dict" not in payload:
        raise ValueError("Full training checkpoint missing model_state_dict.")
    return payload


def _to_plain_dict(value) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {"value": value}


def _resolve_step(
    explicit: int | None,
    trainer_state: dict[str, Any],
    metadata: dict[str, Any],
) -> int:
    value = explicit
    if value is None:
        value = trainer_state.get("global_step", metadata.get("global_step", 0))
    if type(value) is not int:
        value = int(value)
    if value < 0:
        raise ValueError("global_step must be non-negative.")
    return value


def _resolve_epoch(
    explicit: int | None,
    trainer_state: dict[str, Any],
    metadata: dict[str, Any],
) -> int | None:
    value = explicit
    if value is None:
        value = trainer_state.get("epoch", metadata.get("epoch"))
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError("epoch must be non-negative.")
    return value


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for full training checkpoint operations."
        ) from exc
    return torch
