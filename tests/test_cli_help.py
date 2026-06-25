from __future__ import annotations

import subprocess
import sys

from mopforge.cli.main import main


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mopforge.cli.main", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_cli_version_and_doctor(capsys) -> None:
    assert main(["version"]) == 0
    assert "0.46.0" in capsys.readouterr().out

    assert main(["doctor", "--root", "outputs/doctor_test"]) == 0
    output = capsys.readouterr().out
    assert "mopforge_version=0.46.0" in output
    assert "torch_available=" in output


def test_major_cli_help_commands() -> None:
    commands = [
        [],
        ["config"],
        ["claim"],
        ["runtime"],
        ["gpu"],
        ["train"],
        ["sft"],
        ["pretrain"],
        ["experiment"],
        ["benchmark"],
        ["analyze"],
        ["dataset"],
        ["model"],
    ]
    for command in commands:
        completed = run_cli(*command, "--help")
        assert completed.returncode == 0, completed.stdout
        assert "usage:" in completed.stdout


def test_gpu_validate_output_distinguishes_execution(capsys) -> None:
    assert main(["gpu", "validate", "configs/jobs/tiny_gpu_smoke.json"]) == 0
    output = capsys.readouterr().out
    assert "validation=valid" in output
    assert "dry_run=available" in output
    assert "executes_training=False" in output


def test_torchrun_dry_run_never_launches(capsys) -> None:
    assert main(["gpu", "launch-torchrun", "configs/jobs/multigpu_mop_torchrun_plan.json", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "executes=False" in output
    assert "dry_run=True" in output
    assert "torchrun" in output
