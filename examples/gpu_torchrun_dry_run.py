"""Show a torchrun dry-run command without executing it."""

from __future__ import annotations

from mopforge.configs import MoPForgeConfig
from mopforge.gpu import DistributedConfig, build_torchrun_command


def main() -> None:
    path = "configs/jobs/multigpu_mop_torchrun_plan.json"
    envelope = MoPForgeConfig.load(path)
    distributed = DistributedConfig.from_dict(envelope.metadata["distributed"])
    print("executes=False")
    print("command=" + " ".join(build_torchrun_command(path, distributed)))


if __name__ == "__main__":
    main()
