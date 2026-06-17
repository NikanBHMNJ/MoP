"""Manage local artifacts and tiny checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from mopforge.artifacts import ArtifactManager, CheckpointManager
from mopforge.models import TinyCausalTransformer
from mopforge.tokenization import ByteTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
COMPARISON_PATH = OUTPUTS / "tiny_comparison_results.json"
MANIFEST_EXPORT_PATH = OUTPUTS / "artifact_manifest_export.json"


def main() -> None:
    """Run a tiny local artifact/checkpoint smoke example."""

    print(
        "CPU smoke artifact/checkpoint MVP only. Checkpoints are tiny local "
        "files, not production model releases."
    )
    manager = ArtifactManager(ARTIFACT_ROOT)
    comparison_path = ensure_json_artifact()
    copied = manager.copy_artifact(
        comparison_path,
        "metrics",
        artifact_id=f"metrics-tiny-comparison-{uuid4().hex[:8]}",
        model_type="tiny_comparison",
        metadata={"example": "manage_artifacts_and_checkpoints"},
    )

    if TinyCausalTransformer is None:
        print("PyTorch is not installed; skipping checkpoint save/load.")
        manifest_path = manager.export_manifest_json(MANIFEST_EXPORT_PATH)
        print(f"registered artifact ID: {copied.artifact_id}")
        print(f"manifest export path: {manifest_path}")
        return

    tokenizer = ByteTokenizer()
    model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
    )
    checkpoint_manager = CheckpointManager(manager)
    checkpoint = checkpoint_manager.save_torch_checkpoint(
        model,
        run_id="example-artifact-run",
        model_type="tiny_dense",
        module="core",
        step=1,
        metadata={"example": "manage_artifacts_and_checkpoints"},
    )

    fresh_model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
    )
    checkpoint_manager.load_torch_checkpoint(fresh_model, checkpoint)
    latest = checkpoint_manager.latest_checkpoint(model_type="tiny_dense", module="core")
    manifest_path = manager.export_manifest_json(MANIFEST_EXPORT_PATH)

    print(f"registered artifact ID: {copied.artifact_id}")
    print(f"checkpoint ID: {checkpoint.artifact_id}")
    print(f"checkpoint path: {checkpoint.path}")
    print(f"latest checkpoint ID: {latest.artifact_id if latest else None}")
    print(f"manifest export path: {manifest_path}")


def ensure_json_artifact() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if COMPARISON_PATH.exists():
        return COMPARISON_PATH
    COMPARISON_PATH.write_text(
        json.dumps(
            [
                {
                    "model": "tiny_dense",
                    "routing": "none",
                    "train_loss_last": 0.0,
                    "eval_loss_mean": 0.0,
                    "finite": True,
                }
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return COMPARISON_PATH


if __name__ == "__main__":
    main()
