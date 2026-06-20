"""GPU research beta APIs for MoP-Forge."""

from mopforge.gpu.checkpointing import (
    load_gpu_checkpoint,
    restore_gpu_checkpoint,
    save_gpu_checkpoint,
)
from mopforge.gpu.activation_cache import (
    CachedActivationDataset,
    build_cached_activation_dataloaders,
    config_hash,
    file_sha256,
    load_activation_cache,
    write_activation_cache,
)
from mopforge.gpu.config import GPUTrainingConfig, GPUTrainingResult, GPUTrainingState
from mopforge.gpu.data import (
    GPUDataConfig,
    StreamingJSONLDataset,
    build_gpu_dataloaders,
    load_gpu_lesson_splits,
)
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
from mopforge.gpu.efficiency_data import prepare_efficiency_dataset
from mopforge.gpu.quality_gates import (
    evaluate_efficiency_gates,
    write_gate_report,
)
from mopforge.gpu.registry import GPURunRecord, GPURunRegistry
from mopforge.gpu.scaler import AmpScaler
from mopforge.gpu.trainer import (
    GPUTrainer,
    apply_activation_checkpointing,
    offload_cached_frozen_backbone,
    select_attention_metadata,
)
from mopforge.gpu.validation import dry_run_gpu_training_config, validate_gpu_training_config
from mopforge.gpu.warm_sparse import (
    DEFAULT_BOTTLENECKS,
    DEFAULT_LEARNING_RATES,
    write_warm_sparse_sweep_configs,
)

__all__ = [
    "AmpScaler",
    "CachedActivationDataset",
    "DistributedConfig",
    "DEFAULT_BOTTLENECKS",
    "DEFAULT_LEARNING_RATES",
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
    "build_cached_activation_dataloaders",
    "build_module_routing_plan",
    "build_torchrun_command",
    "config_hash",
    "dry_run_gpu_training_config",
    "estimate_active_parameters",
    "estimate_from_config",
    "estimate_training_memory",
    "evaluate_efficiency_gates",
    "fast_parameter_metadata",
    "file_sha256",
    "group_batch_by_modules",
    "launch_torchrun_dry_run",
    "load_activation_cache",
    "load_gpu_checkpoint",
    "load_gpu_lesson_splits",
    "model_profile_100m_dense",
    "model_profile_100m_mop",
    "offload_cached_frozen_backbone",
    "model_profile_1b_mop",
    "model_profile_2b_mop",
    "model_profile_500m_dense",
    "model_profile_500m_mop",
    "model_profile_7b_mop",
    "prepare_efficiency_dataset",
    "restore_gpu_checkpoint",
    "routing_density",
    "save_gpu_checkpoint",
    "select_attention_metadata",
    "validate_distributed_plan",
    "validate_gpu_training_config",
    "write_activation_cache",
    "write_gate_report",
    "write_memory_estimate",
    "write_warm_sparse_sweep_configs",
]
