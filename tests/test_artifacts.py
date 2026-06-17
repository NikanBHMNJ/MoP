"""Tests for local artifact and checkpoint management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.artifacts import ArtifactManager, ArtifactRecord, CheckpointManager
from mopforge.models import TinyCausalTransformer
from mopforge.tokenization import ByteTokenizer


def tiny_model():
    if TinyCausalTransformer is None:
        pytest.skip("PyTorch is not installed.")
    tokenizer = ByteTokenizer()
    return TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
    )


def test_artifact_record_dict_round_trip() -> None:
    record = ArtifactRecord(
        artifact_id="artifact-1",
        kind="metrics",
        path="outputs/metrics.json",
        run_id="run-1",
        model_type="tiny_dense",
        module="core",
        step=1,
        created_at="2026-01-01T00:00:00+00:00",
        metadata={"finite": True},
    )

    loaded = ArtifactRecord.from_dict(record.to_dict())

    assert loaded == record


def test_invalid_artifact_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        ArtifactRecord(
            artifact_id="bad",
            kind="weights",
            path="bad.bin",
        )


def test_artifact_manager_register_writes_manifest(tmp_path) -> None:
    manager = ArtifactManager(tmp_path / "artifacts")
    record = ArtifactRecord(
        artifact_id="metrics-1",
        kind="metrics",
        path="outputs/metrics.json",
    )

    manager.register(record)

    assert manager.manifest_path.exists()
    assert manager.get("metrics-1") == record
    assert manager.exists("metrics-1") is True


def test_duplicate_artifact_ids_are_rejected(tmp_path) -> None:
    manager = ArtifactManager(tmp_path / "artifacts")
    record = ArtifactRecord(
        artifact_id="metrics-1",
        kind="metrics",
        path="outputs/metrics.json",
    )
    manager.register(record)

    with pytest.raises(ValueError, match="Duplicate"):
        manager.register(record)


def test_list_filters_by_kind_run_and_module(tmp_path) -> None:
    manager = ArtifactManager(tmp_path / "artifacts")
    manager.register(
        ArtifactRecord(
            artifact_id="checkpoint-1",
            kind="checkpoint",
            path="a.pt",
            run_id="run-1",
            model_type="tiny_dense",
            module="core",
            step=1,
        )
    )
    manager.register(
        ArtifactRecord(
            artifact_id="metrics-1",
            kind="metrics",
            path="metrics.json",
            run_id="run-2",
            model_type="tiny_mop",
            module="debugging",
        )
    )

    records = manager.list(kind="checkpoint", run_id="run-1", module="core")

    assert [record.artifact_id for record in records] == ["checkpoint-1"]


def test_copy_artifact_copies_file_and_registers(tmp_path) -> None:
    manager = ArtifactManager(tmp_path / "artifacts")
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}', encoding="utf-8")

    record = manager.copy_artifact(
        source,
        "metrics",
        run_id="run-1",
        metadata={"name": "source"},
    )

    assert Path(record.path).exists()
    assert Path(record.path).read_text(encoding="utf-8") == '{"ok": true}'
    assert record.kind == "metrics"
    assert record.metadata["name"] == "source"


def test_manifest_export_works(tmp_path) -> None:
    manager = ArtifactManager(tmp_path / "artifacts")
    manager.register(
        ArtifactRecord(
            artifact_id="metrics-1",
            kind="metrics",
            path="metrics.json",
        )
    )

    path = manager.export_manifest_json(tmp_path / "manifest.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded[0]["artifact_id"] == "metrics-1"


def test_checkpoint_manager_saves_tiny_model_if_torch_installed(tmp_path) -> None:
    model = tiny_model()
    manager = ArtifactManager(tmp_path / "artifacts")
    checkpoints = CheckpointManager(manager)

    record = checkpoints.save_torch_checkpoint(
        model,
        run_id="run-1",
        model_type="tiny_dense",
        module="core",
        step=1,
    )

    assert record.kind == "checkpoint"
    assert Path(record.path).exists()
    assert manager.get(record.artifact_id) == record


def test_checkpoint_manager_loads_into_fresh_model_if_torch_installed(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    model = tiny_model()
    manager = ArtifactManager(tmp_path / "artifacts")
    checkpoints = CheckpointManager(manager)
    record = checkpoints.save_torch_checkpoint(
        model,
        model_type="tiny_dense",
        module="core",
        step=1,
    )

    loaded_model = tiny_model()
    payload = checkpoints.load_torch_checkpoint(loaded_model, record)

    assert "state_dict" in payload
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded_model.state_dict()[key])


def test_latest_checkpoint_returns_highest_step(tmp_path) -> None:
    model = tiny_model()
    manager = ArtifactManager(tmp_path / "artifacts")
    checkpoints = CheckpointManager(manager)
    first = checkpoints.save_torch_checkpoint(
        model,
        model_type="tiny_dense",
        module="core",
        step=1,
    )
    second = checkpoints.save_torch_checkpoint(
        model,
        model_type="tiny_dense",
        module="core",
        step=2,
    )

    latest = checkpoints.latest_checkpoint(model_type="tiny_dense", module="core")

    assert first.step == 1
    assert latest == second


def test_package_imports_without_forcing_cuda() -> None:
    try:
        import torch
    except Exception:
        assert ArtifactManager is not None
        return

    assert torch.cuda.is_available() is False
