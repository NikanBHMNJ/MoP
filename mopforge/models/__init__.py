"""Model smoke-test baselines for MoP-Forge."""

from mopforge.models.fast_adapters import (
    FastAdapter,
    FastAdapterBank,
    FastAdapterConfig,
    adapter_names_from_target_modules,
    normalize_adapter_names,
)
from mopforge.models.generated_params import (
    ConditionEmbedding,
    GeneratedAdapter,
    GeneratedParameterConfig,
    condition_names_from_target_modules,
    normalize_condition_names,
)
from mopforge.models.architectures import (
    ModelArchitectureConfig,
    build_tiny_model_from_architecture,
    parameter_summary_for_architecture,
)
from mopforge.models.manifest import ModelConfig, ModelManifest
from mopforge.models.registry import ModelRecord, ModelRegistry
from mopforge.models.tiny_dense import TinyCausalTransformer
from mopforge.models.tiny_mop import TinyMoPCausalTransformer
from mopforge.models.tiny_router import TinyModuleRouter, predict_modules
from mopforge.models.production_decoder import (
    ProductionCausalLM,
    ProductionDecoderConfig,
    RMSNorm,
    production_parameter_count,
)
from mopforge.models.hf_export import (
    export_gpu_checkpoint_to_huggingface,
    export_huggingface_llama,
)
from mopforge.models.checkpoint_loader import (
    architecture_from_gpu_config,
    load_gpu_checkpoint_model,
)

__all__ = [
    "FastAdapter",
    "FastAdapterBank",
    "FastAdapterConfig",
    "ConditionEmbedding",
    "GeneratedAdapter",
    "GeneratedParameterConfig",
    "ModelArchitectureConfig",
    "ModelConfig",
    "ModelManifest",
    "ModelRecord",
    "ModelRegistry",
    "TinyCausalTransformer",
    "TinyMoPCausalTransformer",
    "TinyModuleRouter",
    "ProductionCausalLM",
    "ProductionDecoderConfig",
    "RMSNorm",
    "adapter_names_from_target_modules",
    "condition_names_from_target_modules",
    "build_tiny_model_from_architecture",
    "normalize_adapter_names",
    "normalize_condition_names",
    "parameter_summary_for_architecture",
    "predict_modules",
    "production_parameter_count",
    "export_huggingface_llama",
    "export_gpu_checkpoint_to_huggingface",
    "architecture_from_gpu_config",
    "load_gpu_checkpoint_model",
]
