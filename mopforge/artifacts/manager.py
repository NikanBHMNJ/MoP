"""File-backed artifact manifest and checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from mopforge.artifacts.schema import ArtifactRecord
from mopforge.lifecycle import CHECKPOINT_FORMAT_VERSION
from mopforge.lifecycle.checkpoint import (
    load_full_training_checkpoint,
    save_full_training_checkpoint,
)


KIND_DIRECTORIES = {
    "checkpoint": "checkpoints",
    "metrics": "metrics",
    "curriculum_plan": "configs",
    "generation_eval": "evaluations",
    "feedback_export": "evaluations",
    "queue_item": "other",
    "config": "configs",
    "other": "other",
}


class ArtifactManager:
    """Local JSONL manifest for experiment artifacts."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        """Create an artifact manager rooted at ``root``."""

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        for directory in set(KIND_DIRECTORIES.values()):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.manifest_path.touch(exist_ok=True)

    def register(self, record: ArtifactRecord) -> ArtifactRecord:
        """Append an artifact record to the manifest."""

        record.validate()
        if self.exists(record.artifact_id):
            raise ValueError(f"Duplicate artifact_id: {record.artifact_id}")
        with self.manifest_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return record

    def list(
        self,
        kind: str | None = None,
        run_id: str | None = None,
        model_type: str | None = None,
        module: str | None = None,
        limit: int | None = None,
    ) -> list[ArtifactRecord]:
        """List artifact records with optional filters."""

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a non-negative integer.")
        records: list[ArtifactRecord] = []
        for record in self._iter_records():
            if kind is not None and record.kind != kind:
                continue
            if run_id is not None and record.run_id != run_id:
                continue
            if model_type is not None and record.model_type != model_type:
                continue
            if module is not None and record.module != module:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        """Return one artifact record by ID."""

        for record in self._iter_records():
            if record.artifact_id == artifact_id:
                return record
        return None

    def exists(self, artifact_id: str) -> bool:
        """Return whether an artifact ID exists in the manifest."""

        return self.get(artifact_id) is not None

    def export_manifest_json(self, path: str | Path) -> Path:
        """Export the JSONL manifest to a JSON array."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                [record.to_dict() for record in self._iter_records()],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output_path

    def copy_artifact(
        self,
        source_path: str | Path,
        kind: str,
        **metadata: Any,
    ) -> ArtifactRecord:
        """Copy a local file under the artifact root and register it."""

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Artifact source does not exist: {source}")

        record_metadata = dict(metadata.pop("metadata", {}) or {})
        known_fields = {
            "artifact_id",
            "run_id",
            "queue_item_id",
            "model_type",
            "module",
            "step",
            "created_at",
        }
        extra_metadata = {
            key: metadata.pop(key)
            for key in list(metadata)
            if key not in known_fields
        }
        record_metadata.update(extra_metadata)

        artifact_id = metadata.pop("artifact_id", None) or _artifact_id(kind, source)
        target_dir = self.root / KIND_DIRECTORIES.get(kind, "other")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{artifact_id}{source.suffix}"
        shutil.copy2(source, target)

        return self.register(
            ArtifactRecord(
                artifact_id=artifact_id,
                kind=kind,
                path=str(target),
                run_id=metadata.pop("run_id", None),
                queue_item_id=metadata.pop("queue_item_id", None),
                model_type=metadata.pop("model_type", None),
                module=metadata.pop("module", None),
                step=metadata.pop("step", None),
                created_at=metadata.pop("created_at", None),
                metadata=record_metadata,
            )
        )

    def _iter_records(self) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        if not self.manifest_path.exists():
            return records
        with self.manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(ArtifactRecord.from_dict(json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in artifact manifest at line {line_number}: "
                        f"{exc.msg}."
                    ) from exc
        return records


class CheckpointManager:
    """Tiny PyTorch state_dict checkpoint helper backed by artifacts."""

    def __init__(self, artifact_manager: ArtifactManager) -> None:
        """Create a checkpoint manager using an artifact manager."""

        self.artifact_manager = artifact_manager

    def save_torch_checkpoint(
        self,
        model,
        path: str | Path | None = None,
        *,
        run_id: str | None = None,
        model_type: str | None = None,
        module: str | None = None,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Save a tiny PyTorch ``state_dict`` checkpoint and register it."""

        torch = _require_torch()
        artifact_id = _checkpoint_artifact_id(model_type, module, step)
        checkpoint_path = Path(path) if path is not None else (
            self.artifact_manager.root / "checkpoints" / f"{artifact_id}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
            "model_type": model_type,
            "module": module,
            "step": step,
        }
        torch.save(payload, checkpoint_path)
        return self.artifact_manager.register(
            ArtifactRecord(
                artifact_id=artifact_id,
                kind="checkpoint",
                path=str(checkpoint_path),
                run_id=run_id,
                model_type=model_type,
                module=module,
                step=step,
                metadata=dict(metadata or {}),
            )
        )

    def load_torch_checkpoint(
        self,
        model,
        artifact_or_path,
        map_location: str = "cpu",
    ) -> dict:
        """Load a PyTorch checkpoint into ``model`` and return its payload."""

        torch = _require_torch()
        path = Path(artifact_or_path.path) if isinstance(
            artifact_or_path, ArtifactRecord
        ) else Path(artifact_or_path)
        payload = torch.load(path, map_location=map_location)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
        return payload if isinstance(payload, dict) else {"state_dict": state_dict}

    def save_full_training_checkpoint(
        self,
        model,
        path: str | Path | None = None,
        *,
        optimizer=None,
        scheduler=None,
        trainer_state=None,
        config=None,
        tokenizer_spec=None,
        parameter_policy=None,
        adapter_metadata=None,
        generated_metadata=None,
        rng_state=None,
        run_id: str | None = None,
        model_type: str | None = None,
        training_kind: str | None = None,
        module: str | None = None,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Save a full training checkpoint and register it in the manifest."""

        metadata_dict = dict(metadata or {})
        artifact_id = _full_checkpoint_artifact_id(
            training_kind=training_kind,
            model_type=model_type,
            module=module,
            step=step,
        )
        checkpoint_path = Path(path) if path is not None else (
            self.artifact_manager.root / "checkpoints" / f"{artifact_id}.pt"
        )
        has_rng_state = not (
            isinstance(rng_state, dict) and rng_state.get("disabled") is True
        )
        metadata_dict.update(
            {
                "full_checkpoint": True,
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "has_optimizer": optimizer is not None,
                "has_scheduler": scheduler is not None,
                "has_rng_state": has_rng_state,
                "global_step": step,
                "training_kind": training_kind,
                "run_id": run_id,
                "model_type": model_type,
                "checkpoint_id": artifact_id,
            }
        )
        record = save_full_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            config=config,
            tokenizer_spec=tokenizer_spec,
            parameter_policy=parameter_policy,
            adapter_metadata=adapter_metadata,
            generated_metadata=generated_metadata,
            rng_state=rng_state,
            metadata=metadata_dict,
            run_id=run_id,
            checkpoint_id=artifact_id,
            global_step=step,
            model_type=model_type,
            training_kind=training_kind,
            artifact_id=artifact_id,
        )
        metadata_dict["record"] = record.to_dict()
        return self.artifact_manager.register(
            ArtifactRecord(
                artifact_id=artifact_id,
                kind="checkpoint",
                path=str(checkpoint_path),
                run_id=run_id,
                model_type=model_type,
                module=module,
                step=step,
                metadata=metadata_dict,
            )
        )

    def load_full_training_checkpoint(
        self,
        artifact_or_path,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load a full training checkpoint payload by artifact record or path."""

        path = Path(artifact_or_path.path) if isinstance(
            artifact_or_path, ArtifactRecord
        ) else Path(artifact_or_path)
        return load_full_training_checkpoint(path, map_location=map_location)

    def latest_checkpoint(
        self,
        model_type: str | None = None,
        module: str | None = None,
        run_id: str | None = None,
    ) -> ArtifactRecord | None:
        """Return the highest-step/latest checkpoint matching filters."""

        checkpoints = self.list_checkpoints(
            model_type=model_type,
            module=module,
            run_id=run_id,
        )
        if not checkpoints:
            return None
        return sorted(
            checkpoints,
            key=lambda record: (
                record.step if record.step is not None else -1,
                record.created_at or "",
                record.artifact_id,
            ),
        )[-1]

    def list_checkpoints(
        self,
        model_type: str | None = None,
        module: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[ArtifactRecord]:
        """List checkpoint artifact records."""

        return self.artifact_manager.list(
            kind="checkpoint",
            run_id=run_id,
            model_type=model_type,
            module=module,
            limit=limit,
        )

    def latest_full_checkpoint(
        self,
        model_type: str | None = None,
        run_id: str | None = None,
        training_kind: str | None = None,
    ) -> ArtifactRecord | None:
        """Return the latest registered full checkpoint matching filters."""

        checkpoints = [
            record
            for record in self.artifact_manager.list(
                kind="checkpoint",
                run_id=run_id,
                model_type=model_type,
            )
            if record.metadata.get("full_checkpoint") is True
            and (
                training_kind is None
                or record.metadata.get("training_kind") == training_kind
            )
        ]
        if not checkpoints:
            return None
        return sorted(
            checkpoints,
            key=lambda record: (
                record.step if record.step is not None else -1,
                record.created_at or "",
                record.artifact_id,
            ),
        )[-1]


def _artifact_id(kind: str, source: Path) -> str:
    digest = hashlib.sha1()
    digest.update(str(source).encode("utf-8", errors="replace"))
    digest.update(source.read_bytes())
    return f"{_slug(kind)}-{_slug(source.stem)}-{digest.hexdigest()[:10]}"


def _checkpoint_artifact_id(
    model_type: str | None,
    module: str | None,
    step: int | None,
) -> str:
    step_part = f"step-{step}" if step is not None else "step-none"
    return (
        f"checkpoint-{_slug(model_type or 'model')}-"
        f"{_slug(module or 'all')}-{step_part}-{uuid4().hex[:8]}"
    )


def _full_checkpoint_artifact_id(
    training_kind: str | None,
    model_type: str | None,
    module: str | None,
    step: int | None,
) -> str:
    step_part = f"step-{step}" if step is not None else "step-none"
    return (
        f"full-checkpoint-{_slug(training_kind or 'training')}-"
        f"{_slug(model_type or 'model')}-{_slug(module or 'all')}-"
        f"{step_part}-{uuid4().hex[:8]}"
    )


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "artifact"


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for checkpoint operations.") from exc
    return torch
