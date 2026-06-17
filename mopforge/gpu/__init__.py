"""GPU research beta APIs for MoP-Forge."""

from mopforge.gpu.checkpointing import (
    load_gpu_checkpoint,
    restore_gpu_checkpoint,
    save_gpu_checkpoint,
)
from mopforge.gpu.config import GPUTrainingConfig, GPUTrainingResult, GPUTrainingState
from mopforge.gpu.data import GPUDataConfig, StreamingJSONLDataset, build_gpu_dataloaders
from mopforge.gpu.distributed import (
    DistributedConfig,
    build_torchrun_command,
    validate_distributed_plan,
)
from mopforge.gpu.launcher import launch_torchrun_dry_run
from mopforge.gpu.memory import (
    ModelMemoryEstimate,
    estimate_from_config,
    estimate_training_memory,
    write_memory_estimate,
)
from mopforge.gpu.mop_execution import (
    ModuleRoutingPlan,
    build_module_routing_plan,
    estimate_active_parameters,
    fast_parameter_metadata,
    group_batch_by_modules,
    routing_density,
)
from mopforge.gpu.profiles import (
    model_profile_100m_dense,
    model_profile_100m_mop,
    model_profile_1b_mop,
    model_profile_2b_mop,
    model_profile_500m_dense,
    model_profile_500m_mop,
    model_profile_7b_mop,
)
from mopforge.gpu.registry import GPURunRecord, GPURunRegistry
from mopforge.gpu.scaler import AmpScaler
from mopforge.gpu.trainer import (
    GPUTrainer,
    apply_activation_checkpointing,
    select_attention_metadata,
)
from mopforge.gpu.validation import dry_run_gpu_training_config, validate_gpu_training_config

__all__ = [
    "AmpScaler",
    "DistributedConfig",
    "GPUDataConfig",
    "GPUTrainer",
    "GPUTrainingConfig",
    "GPUTrainingResult",
    "GPUTrainingState",
    "GPURunRecord",
    "GPURunRegistry",
    "ModelMemoryEstimate",
    "ModuleRoutingPlan",
    "StreamingJSONLDataset",
    "apply_activation_checkpointing",
    "build_gpu_dataloaders",
    "build_module_routing_plan",
    "build_torchrun_command",
    "dry_run_gpu_training_config",
    "estimate_active_parameters",
    "estimate_from_config",
    "estimate_training_memory",
    "fast_parameter_metadata",
    "group_batch_by_modules",
    "launch_torchrun_dry_run",
    "load_gpu_checkpoint",
    "model_profile_100m_dense",
    "model_profile_100m_mop",
    "model_profile_1b_mop",
    "model_profile_2b_mop",
    "model_profile_500m_dense",
    "model_profile_500m_mop",
    "model_profile_7b_mop",
    "restore_gpu_checkpoint",
    "routing_density",
    "save_gpu_checkpoint",
    "select_attention_metadata",
    "validate_distributed_plan",
    "validate_gpu_training_config",
    "write_memory_estimate",
]
