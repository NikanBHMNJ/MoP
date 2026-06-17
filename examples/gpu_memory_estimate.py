"""Print memory estimates for GPU job profiles."""

from __future__ import annotations

from mopforge.configs import get_default_config
from mopforge.gpu import estimate_from_config
from mopforge.configs import gpu_training_config_from_envelope


def main() -> None:
    for name in ("gpu_tiny_smoke", "gpu_100m_mop_a100", "gpu_2b_mop_a100_plan"):
        config = gpu_training_config_from_envelope(get_default_config(name))
        estimate = estimate_from_config(config)
        print(f"{name}: total_gb={estimate.total_memory_gb_estimate} fits={estimate.fits}")


if __name__ == "__main__":
    main()
