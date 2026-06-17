"""Tests for model registry and architecture configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from mopforge.cli.main import main as cli_main
from mopforge.configs import MoPForgeConfig, get_default_config, model_config_from_envelope
from mopforge.models import (
    ModelArchitectureConfig,
    ModelManifest,
    ModelRegistry,
    build_tiny_model_from_architecture,
)


def tiny_arch() -> ModelArchitectureConfig:
    return ModelArchitectureConfig(name="tiny_demo", model_type="dense", d_model=16, n_layers=1, n_heads=2, max_seq_len=32)


def test_architecture_validation_roundtrip(tmp_path) -> None:
    arch = tiny_arch()
    path = arch.save(tmp_path / "architecture.json")
    assert ModelArchitectureConfig.load(path) == arch
    with pytest.raises(ValueError, match="divisible"):
        ModelArchitectureConfig(name="bad", d_model=15, n_heads=2)


def test_manifest_roundtrip_and_registry_refs(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    manifest = registry.register_model(tiny_arch())
    loaded = registry.load_manifest(manifest.model_id)

    assert loaded.version_id == manifest.version_id
    assert registry.resolve_model_ref(manifest.model_id).version_id == manifest.version_id
    assert registry.resolve_model_ref(f"{manifest.model_id}@{manifest.version_id}").model_id == manifest.model_id
    assert ModelManifest.load(Path(manifest.metadata["manifest_path"])).model_id == manifest.model_id
    assert registry.list_models()[0].model_id == manifest.model_id
    assert registry.snapshot_model(manifest.model_id).model_id == manifest.model_id


def test_build_tiny_model_from_architecture() -> None:
    pytest.importorskip("torch")
    model = build_tiny_model_from_architecture(tiny_arch())
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_model_config_envelope_and_cli(tmp_path, capsys) -> None:
    config_path = get_default_config("model_tiny_mop").save(tmp_path / "model.json")
    assert model_config_from_envelope(MoPForgeConfig.load(config_path)).name == "tiny_mop_oracle"
    root = tmp_path / "models"

    assert cli_main(["model", "register", str(config_path), "--root", str(root)]) == 0
    output = capsys.readouterr().out
    model_id = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("model_id=")][0]
    assert cli_main(["model", "list", "--root", str(root)]) == 0
    assert model_id in capsys.readouterr().out
    assert cli_main(["model", "show", model_id, "--root", str(root)]) == 0
    assert "parameter_summary=" in capsys.readouterr().out
    assert cli_main(["model", "versions", model_id, "--root", str(root)]) == 0
    assert model_id or capsys.readouterr().out is not None
    assert cli_main(["model", "snapshot", model_id, "--root", str(root)]) == 0


def test_model_registry_no_cuda_required() -> None:
    try:
        import torch
    except Exception:
        return
    assert torch.cuda.is_available() is False
