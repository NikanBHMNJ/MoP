"""Tests for research run manifests."""

from __future__ import annotations

from mopforge.cli.main import main as cli_main
from mopforge.configs import default_sft_config
from mopforge.manifests import ManifestRegistry, ResourceSpec, command_text, plan_run_manifest
from mopforge.manifests.run_manifest import ResearchRunManifest


def test_resource_validation() -> None:
    assert ResourceSpec(accelerator="cpu").num_gpus == 0
    try:
        ResourceSpec(accelerator="a100_80gb")
        assert False
    except ValueError:
        pass


def test_manifest_roundtrip_registry_and_planner(tmp_path) -> None:
    manifest = plan_run_manifest(default_sft_config("sft_full"), ResourceSpec(accelerator="cpu"), name="demo", config_ref="demo.json")
    assert "mopforge sft run demo.json" == command_text(manifest)
    path = manifest.save(tmp_path / "manifest.json")
    assert ResearchRunManifest.load(path).manifest_id == manifest.manifest_id
    registry = ManifestRegistry(tmp_path / "manifests")
    registry.create(manifest)
    assert registry.load(manifest.manifest_id).name == "demo"
    assert registry.export_command(manifest.manifest_id).exists()


def test_manifest_cli(tmp_path, capsys) -> None:
    config_path = default_sft_config("sft_full").save(tmp_path / "sft.json")
    root = tmp_path / "manifests"
    assert cli_main(["manifest", "create", str(config_path), "--name", "smoke", "--accelerator", "cpu", "--root", str(root)]) == 0
    output = capsys.readouterr().out
    manifest_id = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("manifest_id=")][0]
    assert cli_main(["manifest", "dry-run", manifest_id, "--root", str(root)]) == 0
    assert "executes_gpu" in capsys.readouterr().out
    assert cli_main(["manifest", "list", "--root", str(root)]) == 0
    assert manifest_id in capsys.readouterr().out
    assert cli_main(["manifest", "show", manifest_id, "--root", str(root)]) == 0
    assert "command=" in capsys.readouterr().out
    assert cli_main(["manifest", "export-command", manifest_id, "--root", str(root)]) == 0
