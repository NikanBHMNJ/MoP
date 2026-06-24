from pathlib import Path

import pytest

from mopforge.gpu import run_gpu_probe
from tests.test_gpu_trainer import _config


pytest.importorskip("torch")


def test_staged_gpu_probe_writes_resume_checked_report_on_cpu_fallback(tmp_path):
    output = tmp_path / "probe.json"
    config = _config(
        tmp_path,
        max_steps=99,
        max_optimizer_steps=1,
        save_full_checkpoints=False,
        save_optimizer_state=False,
    )

    result = run_gpu_probe(config, optimizer_updates=1, output_path=output)

    assert result["status"] == "completed"
    assert result["training_probe"]["optimizer_updates"] == 1
    assert result["checkpoint_resume_probe"]["passed"] is True
    assert result["automatic_config_changes"] == []
    assert output.exists()
    phase_names = {phase["name"] for phase in result["phases"]}
    assert {
        "model_and_data_allocation",
        "forward",
        "backward",
        "optimizer_state_and_steady_updates",
        "evaluation",
        "atomic_model_only_checkpoint_save",
        "checkpoint_load_and_resume_consistency",
        "cleanup",
    }.issubset(phase_names)
    assert Path(result["checkpoint_resume_probe"]["path"]).exists()
