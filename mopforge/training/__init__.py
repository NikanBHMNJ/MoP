"""Training and routing utilities for MoP-Forge."""

from mopforge.training.parameter_policy import (
    SUPPORTED_POLICY_MODES,
    ParameterGroupSummary,
    TrainableParameterPolicy,
    apply_trainable_policy,
    build_optimizer_for_trainable_parameters,
    count_parameters,
    infer_parameter_group,
    policy_from_queue_item,
    summarize_parameter_groups,
)
from mopforge.training.routing import (
    DEFAULT_KNOWN_MODULES,
    module_mask_from_targets,
    normalize_target_modules,
    route_batch_with_router,
)
from mopforge.training.runner import run_tiny_training_from_curriculum
from mopforge.training.state import TrainerConfig, TrainerResult, TrainerState
from mopforge.training.trainer import TinyTrainer

__all__ = [
    "DEFAULT_KNOWN_MODULES",
    "SUPPORTED_POLICY_MODES",
    "ParameterGroupSummary",
    "TinyTrainer",
    "TrainableParameterPolicy",
    "TrainerConfig",
    "TrainerResult",
    "TrainerState",
    "apply_trainable_policy",
    "build_optimizer_for_trainable_parameters",
    "count_parameters",
    "infer_parameter_group",
    "module_mask_from_targets",
    "normalize_target_modules",
    "policy_from_queue_item",
    "route_batch_with_router",
    "run_tiny_training_from_curriculum",
    "summarize_parameter_groups",
]
