from pathlib import Path

import pytest

from mopforge.cli.main import main
from mopforge.configs import MoPForgeConfig
from tests.test_gpu_trainer import _config


pytest.importorskip("torch")


def test_gpu_cli_validate_estimate_train_list_show_resume_and_launch(tmp_path):
    config = _config(tmp_path, name="cli_gpu", max_steps=1)
    path = tmp_path / "gpu.json"
    MoPForgeConfig(kind="gpu_train", payload=config.to_dict()).save(path)
    assert main(["gpu", "validate", str(path)]) == 0
    assert main(["gpu", "estimate", str(path)]) == 0
    assert main(["gpu", "train", str(path)]) == 0
    records = list((tmp_path / "gpu_runs").iterdir())
    run_dirs = [item for item in records if item.is_dir()]
    assert run_dirs
    run_id = run_dirs[0].name
    assert main(["gpu", "list", "--root", str(tmp_path / "gpu_runs")]) == 0
    assert main(["gpu", "show", run_id, "--root", str(tmp_path / "gpu_runs")]) == 0
    checkpoint = next((run_dirs[0] / "checkpoints").glob("*.pt"))
    assert main(["gpu", "resume", str(checkpoint)]) == 0


def test_gpu_cli_write_default_and_launch_torchrun(tmp_path):
    path = tmp_path / "gpu_tiny.json"
    assert main(["config", "write-default", "gpu_tiny_smoke", str(path)]) == 0
    assert path.exists()
    assert main(["gpu", "launch-torchrun", "configs/jobs/multigpu_mop_torchrun_plan.json", "--dry-run"]) == 0
