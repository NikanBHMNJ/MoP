"""Local CPU ablation framework."""

from mopforge.ablations.config import AblationConfig, AblationVariant
from mopforge.ablations.registry import AblationRecord, AblationRegistry
from mopforge.ablations.runner import (
    AblationResult,
    dry_run_ablation,
    expand_ablation_variants,
    run_ablation,
)

__all__ = [
    "AblationConfig",
    "AblationRecord",
    "AblationRegistry",
    "AblationResult",
    "AblationVariant",
    "dry_run_ablation",
    "expand_ablation_variants",
    "run_ablation",
]
