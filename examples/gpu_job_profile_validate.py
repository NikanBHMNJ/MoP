"""Validate GPU job profiles without executing large training."""

from __future__ import annotations

from mopforge.configs import MoPForgeConfig, gpu_training_config_from_envelope
from mopforge.gpu import validate_gpu_training_config


def main() -> None:
    for path in (
        "configs/jobs/tiny_gpu_smoke.json",
        "configs/jobs/100m_mop_a100_smoke.json",
        "configs/jobs/2b_mop_a100_plan.json",
    ):
        config = gpu_training_config_from_envelope(MoPForgeConfig.load(path))
        messages = validate_gpu_training_config(config)
        print(f"{path}: {messages or ['valid']}")


if __name__ == "__main__":
    main()
