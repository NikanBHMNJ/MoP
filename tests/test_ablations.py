"""Tests for ablation configs and runner."""

from __future__ import annotations

from mopforge.ablations import AblationConfig, AblationRegistry, AblationVariant, dry_run_ablation, expand_ablation_variants
from mopforge.cli.main import main as cli_main
from mopforge.configs import default_sft_config, get_default_config


def test_variant_override_application() -> None:
    config = AblationConfig(name="demo", base_config=default_sft_config("sft_full").to_dict(), variants=[AblationVariant("mop", overrides={"model_type": "mop_oracle"})])
    runs = expand_ablation_variants(config)
    assert runs[0].payload["model_type"] == "mop_oracle"
    assert dry_run_ablation(config)["variant_count"] == 1


def test_ablation_cli_dry_run_and_registry(tmp_path, capsys) -> None:
    path = get_default_config("ablation_adapter_vs_generated").save(tmp_path / "ablation.json")
    assert cli_main(["ablation", "dry-run", str(path)]) == 0
    assert "variant_count" in capsys.readouterr().out
    registry = AblationRegistry(tmp_path / "ablations")
    record = registry.create("demo")
    assert registry.load(record.ablation_id).name == "demo"
