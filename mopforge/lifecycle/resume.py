"""Filesystem checkpoint resolution helpers for CLI resume commands."""

from __future__ import annotations

from pathlib import Path

from mopforge.artifacts import ArtifactManager, CheckpointManager
from mopforge.lifecycle.checkpoint import load_full_training_checkpoint


def resolve_full_checkpoint_reference(
    reference: str,
    *,
    artifact_root: str | Path = "artifacts",
    training_kind: str | None = None,
    model_type: str | None = None,
) -> Path:
    """Resolve a file path, artifact id, or run id to a full checkpoint path."""

    candidate = Path(reference)
    if candidate.exists():
        return candidate

    artifact_manager = ArtifactManager(artifact_root)
    artifact = artifact_manager.get(reference)
    if artifact is not None:
        if not artifact.metadata.get("full_checkpoint"):
            raise ValueError(f"Artifact is not a full checkpoint: {reference}")
        return Path(artifact.path)

    checkpoint = CheckpointManager(artifact_manager).latest_full_checkpoint(
        run_id=reference,
        model_type=model_type,
        training_kind=training_kind,
    )
    if checkpoint is not None:
        return Path(checkpoint.path)
    raise FileNotFoundError(
        "Could not resolve full checkpoint path, artifact id, or run id: "
        f"{reference}"
    )


def load_config_snapshot_from_checkpoint(path: str | Path) -> dict:
    """Load the config snapshot dictionary from a full checkpoint."""

    payload = load_full_training_checkpoint(path)
    config = payload.get("config")
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Checkpoint has no config snapshot: {path}")
    return dict(config)
