"""Tiny experiment harnesses for MoP-Forge."""

from mopforge.experiments.config import TinyExperimentConfig
from mopforge.experiments.matrix import (
    ExperimentConfig,
    RUNNABLE_CONFIG_KINDS,
    expand_experiment_matrix,
)
from mopforge.experiments.registry import ExperimentRecord, ExperimentRegistry
from mopforge.experiments.runner import ExperimentRunResult, run_experiment
from mopforge.experiments.tiny_compare import (
    eval_tiny_dense,
    eval_tiny_mop_learned_router,
    eval_tiny_mop_oracle,
    load_or_generate_lessons,
    run_tiny_comparison,
    train_tiny_dense,
    train_tiny_mop_learned_router,
    train_tiny_mop_oracle,
    train_tiny_router,
    write_results,
)
from mopforge.experiments.utils import set_seed, split_lessons

__all__ = [
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentRunResult",
    "RUNNABLE_CONFIG_KINDS",
    "TinyExperimentConfig",
    "expand_experiment_matrix",
    "eval_tiny_dense",
    "eval_tiny_mop_learned_router",
    "eval_tiny_mop_oracle",
    "load_or_generate_lessons",
    "run_tiny_comparison",
    "run_experiment",
    "set_seed",
    "split_lessons",
    "train_tiny_dense",
    "train_tiny_mop_learned_router",
    "train_tiny_mop_oracle",
    "train_tiny_router",
    "write_results",
]
