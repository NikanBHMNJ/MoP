"""Baseline catalog and comparison helpers."""

from mopforge.baselines.catalog import get_baseline, list_baselines
from mopforge.baselines.config import BaselineConfig, BaselineSpec


def build_baseline_experiment_config(*args, **kwargs):
    from mopforge.baselines.runner import build_baseline_experiment_config as _build

    return _build(*args, **kwargs)

__all__ = [
    "BaselineConfig",
    "BaselineSpec",
    "build_baseline_experiment_config",
    "get_baseline",
    "list_baselines",
]
