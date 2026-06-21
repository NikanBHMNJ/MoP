"""Model architecture configs and tiny construction bridge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODEL_TYPES = {"dense", "mop_oracle", "mop_learned_router", "baseline_moe", "future_large"}
GENERATED_TYPES = {"low_rank_adapter", "scale_shift"}
INTENDED_SCALES = {"tiny_cpu", "small_gpu", "medium_gpu", "large_gpu"}
MOP_BLOCK_TYPES = {"post_core_mlp", "routed_ffn"}
ROUTING_GRANULARITIES = {"example", "token"}


@dataclass(slots=True)
class ModelArchitectureConfig:
    """Serializable architecture description for local model registry entries."""

    name: str
    model_type: str = "dense"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    max_seq_len: int = 512
    vocab_size: int | None = None
    module_names: list[str] = field(default_factory=lambda: ["core", "coding", "debugging", "repair"])
    always_include_core: bool = True
    mop_block_type: str = "post_core_mlp"
    expert_count: int | None = None
    active_experts: int = 1
    routing_granularity: str = "example"
    shared_depth_ratio: float = 1.0
    use_lora_deltas: bool = False
    lora_tail_only: bool = False
    lora_rank: int = 0
    lora_target_modules: list[str] | None = None
    use_fast_adapters: bool = False
    fast_adapter_names: list[str] | None = None
    fast_adapter_bottleneck_dim: int = 16
    use_generated_params: bool = False
    generated_condition_names: list[str] | None = None
    generated_condition_dim: int = 16
    generated_rank: int = 4
    generated_type: str = "low_rank_adapter"
    router_enabled: bool = False
    router_type: str | None = None
    tokenizer_ref: str | None = None
    dataset_ref: str | None = None
    intended_scale: str = "tiny_cpu"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _non_empty(self.name, "name")
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(sorted(MODEL_TYPES))}.")
        for field_name in ("d_model", "n_layers", "n_heads", "max_seq_len"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.vocab_size is not None and (type(self.vocab_size) is not int or self.vocab_size <= 0):
            raise ValueError("vocab_size must be a positive integer or None.")
        self.module_names = _strings(self.module_names, "module_names")
        if not isinstance(self.always_include_core, bool):
            raise ValueError("always_include_core must be a boolean.")
        if self.mop_block_type not in MOP_BLOCK_TYPES:
            raise ValueError("mop_block_type must be post_core_mlp or routed_ffn.")
        if self.expert_count is not None and (type(self.expert_count) is not int or self.expert_count <= 0):
            raise ValueError("expert_count must be a positive integer or None.")
        if type(self.active_experts) is not int or self.active_experts <= 0:
            raise ValueError("active_experts must be a positive integer.")
        if self.routing_granularity not in ROUTING_GRANULARITIES:
            raise ValueError("routing_granularity must be example or token.")
        if type(self.shared_depth_ratio) not in {float, int} or not 0.0 < float(self.shared_depth_ratio) <= 1.0:
            raise ValueError("shared_depth_ratio must be in (0.0, 1.0].")
        self.shared_depth_ratio = float(self.shared_depth_ratio)
        if not isinstance(self.use_lora_deltas, bool):
            raise ValueError("use_lora_deltas must be a boolean.")
        if not isinstance(self.lora_tail_only, bool):
            raise ValueError("lora_tail_only must be a boolean.")
        if self.lora_tail_only and not self.use_lora_deltas:
            raise ValueError("lora_tail_only requires use_lora_deltas=true.")
        if type(self.lora_rank) is not int or self.lora_rank < 0:
            raise ValueError("lora_rank must be a non-negative integer.")
        if self.use_lora_deltas and self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive when use_lora_deltas is true.")
        self.lora_target_modules = _optional_strings(self.lora_target_modules, "lora_target_modules")
        self.fast_adapter_names = _optional_strings(self.fast_adapter_names, "fast_adapter_names")
        self.generated_condition_names = _optional_strings(
            self.generated_condition_names,
            "generated_condition_names",
        )
        if type(self.fast_adapter_bottleneck_dim) is not int or self.fast_adapter_bottleneck_dim <= 0:
            raise ValueError("fast_adapter_bottleneck_dim must be a positive integer.")
        if type(self.generated_condition_dim) is not int or self.generated_condition_dim <= 0:
            raise ValueError("generated_condition_dim must be a positive integer.")
        if type(self.generated_rank) is not int or self.generated_rank <= 0:
            raise ValueError("generated_rank must be a positive integer.")
        if self.generated_type not in GENERATED_TYPES:
            raise ValueError(f"generated_type must be one of: {', '.join(sorted(GENERATED_TYPES))}.")
        if self.router_type is not None:
            self.router_type = _non_empty(self.router_type, "router_type")
        if self.tokenizer_ref is not None:
            self.tokenizer_ref = _non_empty(self.tokenizer_ref, "tokenizer_ref")
        if self.dataset_ref is not None:
            self.dataset_ref = _non_empty(self.dataset_ref, "dataset_ref")
        if self.intended_scale not in INTENDED_SCALES:
            raise ValueError(f"intended_scale must be one of: {', '.join(sorted(INTENDED_SCALES))}.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArchitectureConfig":
        if not isinstance(data, dict):
            raise TypeError("ModelArchitectureConfig.from_dict expects a dictionary.")
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ModelArchitectureConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_tiny_model_from_architecture(config: ModelArchitectureConfig, tokenizer=None):
    """Build a tiny CPU model from an architecture config.

    ``baseline_moe`` is represented by a tiny MoP-backed shim for this MVP.
    ``future_large`` is registry-only and intentionally not instantiated.
    """

    config = ModelArchitectureConfig.from_dict(config.to_dict())
    if config.model_type == "future_large":
        raise ValueError("future_large architectures are registry-only in this MVP.")
    vocab_size = config.vocab_size
    if vocab_size is None:
        if tokenizer is not None:
            from mopforge.tokenization import get_tokenizer_vocab_size

            vocab_size = get_tokenizer_vocab_size(tokenizer)
        else:
            vocab_size = 259
    kwargs = {
        "vocab_size": vocab_size,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "max_seq_len": config.max_seq_len,
        "use_fast_adapters": config.use_fast_adapters,
        "fast_adapter_names": config.fast_adapter_names,
        "fast_adapter_bottleneck_dim": config.fast_adapter_bottleneck_dim,
        "use_generated_params": config.use_generated_params,
        "generated_condition_names": config.generated_condition_names,
        "generated_condition_dim": config.generated_condition_dim,
        "generated_rank": config.generated_rank,
        "generated_type": config.generated_type,
    }
    if config.model_type == "dense":
        from mopforge.models.tiny_dense import TinyCausalTransformer

        if TinyCausalTransformer is None:
            raise RuntimeError("PyTorch is required for TinyCausalTransformer.")
        return TinyCausalTransformer(**kwargs)
    from mopforge.models.tiny_mop import TinyMoPCausalTransformer

    if TinyMoPCausalTransformer is None:
        raise RuntimeError("PyTorch is required for TinyMoPCausalTransformer.")
    return TinyMoPCausalTransformer(
        module_names=config.module_names,
        always_include_core=config.always_include_core,
        mop_block_type=config.mop_block_type,
        expert_count=config.expert_count,
        active_experts=config.active_experts,
        routing_granularity=config.routing_granularity,
        shared_depth_ratio=config.shared_depth_ratio,
        use_lora_deltas=config.use_lora_deltas,
        lora_tail_only=config.lora_tail_only,
        lora_rank=config.lora_rank,
        lora_target_modules=config.lora_target_modules,
        **kwargs,
    )


def parameter_summary_for_architecture(config: ModelArchitectureConfig) -> dict[str, Any]:
    """Best-effort parameter summary without making PyTorch mandatory."""

    try:
        model = build_tiny_model_from_architecture(config)
    except Exception as exc:
        return {
            "instantiated": False,
            "error": str(exc),
            "model_type": config.model_type,
            "intended_scale": config.intended_scale,
        }
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "instantiated": True,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "model_type": config.model_type,
        "intended_scale": config.intended_scale,
    }


def _strings(values: list[str], field_name: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list of strings.")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    seen = set()
    return [value.strip() for value in values if not (value.strip() in seen or seen.add(value.strip()))]


def _optional_strings(values: list[str] | None, field_name: str) -> list[str] | None:
    if values is None:
        return None
    return _strings(values, field_name)


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
