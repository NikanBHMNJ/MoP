"""Tests for baseline catalog."""

from __future__ import annotations

from mopforge.baselines import build_baseline_experiment_config, get_baseline, list_baselines
from mopforge.cli.main import main as cli_main


def test_baseline_catalog_and_experiment(capsys) -> None:
    assert get_baseline("dense_full").family == "dense"
    assert any(spec.name == "moe_tiny" for spec in list_baselines())
    experiment = build_baseline_experiment_config(["dense_full", "adapter_only", "moe_tiny"])
    assert len(experiment.runs) == 3
    assert cli_main(["baseline", "list"]) == 0
    assert "dense_full" in capsys.readouterr().out
    assert cli_main(["baseline", "show", "moe_tiny"]) == 0
    assert "moe_tiny_shim" in capsys.readouterr().out
    assert cli_main(["baseline", "experiment", "--baselines", "dense_full", "adapter_only"]) == 0
