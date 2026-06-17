"""Launcher helpers for GPU job plans."""

from __future__ import annotations

from mopforge.gpu.distributed import DistributedConfig, build_torchrun_command


def launch_torchrun_dry_run(config_path: str, distributed: DistributedConfig | None = None) -> dict:
    """Return a dry-run torchrun command payload without executing it."""

    config = distributed or DistributedConfig(strategy="torchrun", dry_run=True)
    return {
        "executes": False,
        "strategy": config.strategy,
        "dry_run": True,
        "command": build_torchrun_command(config_path, config),
    }
