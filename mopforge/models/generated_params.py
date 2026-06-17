"""Tiny generated-parameter adapters for CPU smoke experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_GENERATOR_TYPES = {"low_rank_adapter", "scale_shift"}

DEFAULT_CONDITION_MODULE_MAP = {
    "coding": "coding",
    "debugging": "debugging",
    "repair": "repair",
    "math": "math",
    "planning": "planning",
    "fast_adapter": "default",
}


@dataclass(slots=True)
class GeneratedParameterConfig:
    """Configuration for a tiny generated-parameter adapter."""

    d_model: int
    condition_dim: int = 32
    rank: int = 4
    generator_hidden_dim: int = 64
    generator_type: str = "low_rank_adapter"
    condition_names: list[str] = field(default_factory=lambda: ["default"])
    residual_scale: float = 1.0
    activation: str = "gelu"

    def __post_init__(self) -> None:
        """Validate tiny generated-parameter settings."""

        for field_name in ("d_model", "condition_dim", "rank", "generator_hidden_dim"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.generator_type not in SUPPORTED_GENERATOR_TYPES:
            valid = ", ".join(sorted(SUPPORTED_GENERATOR_TYPES))
            raise ValueError(f"generator_type must be one of: {valid}.")
        if self.activation not in {"gelu", "relu", "tanh"}:
            raise ValueError("activation must be gelu, relu, or tanh.")
        if not isinstance(self.condition_names, list):
            self.condition_names = list(self.condition_names)
        if not self.condition_names or not all(
            isinstance(name, str) and name.strip()
            for name in self.condition_names
        ):
            raise ValueError("condition_names must contain non-empty strings.")
        seen = set()
        normalized = []
        for name in self.condition_names:
            if name in seen:
                raise ValueError("condition_names must be unique.")
            seen.add(name)
            normalized.append(name)
        self.condition_names = normalized
        if not math.isfinite(float(self.residual_scale)):
            raise ValueError("residual_scale must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return {
            "d_model": self.d_model,
            "condition_dim": self.condition_dim,
            "rank": self.rank,
            "generator_hidden_dim": self.generator_hidden_dim,
            "generator_type": self.generator_type,
            "condition_names": list(self.condition_names),
            "residual_scale": float(self.residual_scale),
            "activation": self.activation,
        }


def normalize_condition_names(
    names,
    known_conditions: Iterable[str] | None = None,
    *,
    include_default: bool = False,
) -> list[str]:
    """Normalize condition names, ignoring unknown names when a known set exists."""

    known_list = list(known_conditions) if known_conditions is not None else None
    known_set = set(known_list) if known_list is not None else None
    if names is None:
        raw_names: list[Any] = []
    elif isinstance(names, str):
        raw_names = [names]
    else:
        raw_names = list(names)

    if include_default:
        raw_names = ["default", *raw_names]

    seen = set()
    normalized = []
    for name in raw_names:
        if not isinstance(name, str) or not name.strip():
            continue
        if known_set is not None and name not in known_set:
            continue
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def condition_names_from_target_modules(target_modules: list[str] | str | None) -> list[str]:
    """Map lesson target modules to generated-parameter condition names."""

    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    mapped = [
        DEFAULT_CONDITION_MODULE_MAP[module]
        for module in target_modules
        if module in DEFAULT_CONDITION_MODULE_MAP
    ]
    return normalize_condition_names(mapped)


try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None
    ConditionEmbedding = None
    GeneratedAdapter = None
else:

    class ConditionEmbedding(nn.Module):
        """Named condition embeddings for generated-parameter adapters."""

        def __init__(self, condition_names: Sequence[str], condition_dim: int) -> None:
            super().__init__()
            if type(condition_dim) is not int or condition_dim <= 0:
                raise ValueError("condition_dim must be a positive integer.")
            names = normalize_condition_names(condition_names)
            if not names:
                raise ValueError("condition_names must contain at least one name.")
            self.condition_names = list(names)
            self.name_to_index = {
                name: index for index, name in enumerate(self.condition_names)
            }
            self.embedding = nn.Embedding(len(self.condition_names), condition_dim)

        def forward(self, active_conditions=None):
            """Return the mean embedding for active condition names, or None."""

            names = normalize_condition_names(
                active_conditions,
                self.condition_names,
            )
            return self.embed_names(names)

        def embed_names(self, condition_names: Sequence[str]):
            """Embed already-normalized names and average them deterministically."""

            names = normalize_condition_names(condition_names, self.condition_names)
            if not names:
                return None
            indices = torch.tensor(
                [self.name_to_index[name] for name in names],
                dtype=torch.long,
                device=self.embedding.weight.device,
            )
            return self.embedding(indices).mean(dim=0)


    class GeneratedAdapter(nn.Module):
        """Apply tiny hypernetwork-generated adapter behavior to hidden states.

        ``active_conditions=None`` applies no generated parameters. For active
        conditions, the stored hypernetwork emits temporary adapter tensors for
        that forward pass; those generated tensors are not persistent module
        parameters.
        """

        def __init__(self, config: GeneratedParameterConfig) -> None:
            super().__init__()
            self.config = config
            self.condition_embedding = ConditionEmbedding(
                config.condition_names,
                config.condition_dim,
            )
            self.condition_names = list(config.condition_names)
            output_dim = self._generated_output_dim()
            self.generator = nn.Sequential(
                nn.Linear(config.condition_dim, config.generator_hidden_dim),
                _activation_module(config.activation),
                nn.Linear(config.generator_hidden_dim, output_dim),
            )
            self.hidden_activation = _activation_module(config.activation)

        def forward(self, hidden_states, active_conditions=None):
            """Apply generated adapter deltas to hidden states."""

            expanded = self.expand_active_conditions(
                active_conditions,
                batch_size=int(hidden_states.shape[0]),
            )
            if not any(expanded):
                return hidden_states

            chunks = []
            for index, condition_names in enumerate(expanded):
                example_hidden = hidden_states[index : index + 1]
                condition_vector = self.condition_embedding.embed_names(condition_names)
                if condition_vector is None:
                    chunks.append(example_hidden)
                else:
                    chunks.append(
                        self._apply_condition(example_hidden, condition_vector)
                    )
            return torch.cat(chunks, dim=0)

        def expand_active_conditions(
            self,
            active_conditions,
            batch_size: int,
        ) -> list[list[str]]:
            """Normalize active conditions into one list per batch row."""

            if active_conditions is None:
                return [[] for _ in range(batch_size)]
            if isinstance(active_conditions, str):
                names = normalize_condition_names(
                    active_conditions,
                    self.condition_names,
                )
                return [names for _ in range(batch_size)]

            active_list = list(active_conditions)
            if not active_list:
                return [[] for _ in range(batch_size)]
            if all(isinstance(item, str) for item in active_list):
                names = normalize_condition_names(active_list, self.condition_names)
                return [names for _ in range(batch_size)]
            if len(active_list) != batch_size:
                raise ValueError(
                    "Per-example active_conditions length must match batch size."
                )
            return [
                normalize_condition_names(condition_names, self.condition_names)
                for condition_names in active_list
            ]

        def generated_parameter_count(self) -> dict[str, int]:
            """Return counts for stored and per-forward generated tensors."""

            condition_params = sum(
                parameter.numel()
                for parameter in self.condition_embedding.parameters()
            )
            hypernetwork_params = sum(
                parameter.numel() for parameter in self.generator.parameters()
            )
            return {
                "stored_trainable_params": int(condition_params + hypernetwork_params),
                "condition_embedding_params": int(condition_params),
                "hypernetwork_params": int(hypernetwork_params),
                "generated_tensors_per_condition": int(self._generated_output_dim()),
            }

        def _generated_output_dim(self) -> int:
            if self.config.generator_type == "scale_shift":
                return self.config.d_model * 2
            return self.config.d_model * self.config.rank * 2

        def _apply_condition(self, hidden_states, condition_vector):
            generated = self.generator(condition_vector)
            if self.config.generator_type == "scale_shift":
                return self._apply_scale_shift(hidden_states, generated)
            return self._apply_low_rank(hidden_states, generated)

        def _apply_low_rank(self, hidden_states, generated):
            d_model = self.config.d_model
            rank = self.config.rank
            split = d_model * rank
            down_weight = generated[:split].view(d_model, rank) / math.sqrt(d_model)
            up_weight = generated[split:].view(rank, d_model) / math.sqrt(rank)
            delta = torch.matmul(self.hidden_activation(torch.matmul(hidden_states, down_weight)), up_weight)
            return hidden_states + float(self.config.residual_scale) * delta

        def _apply_scale_shift(self, hidden_states, generated):
            d_model = self.config.d_model
            scale = torch.tanh(generated[:d_model]).view(1, 1, d_model)
            shift = generated[d_model:].view(1, 1, d_model)
            delta = hidden_states * scale + shift
            return hidden_states + float(self.config.residual_scale) * delta


    def _activation_module(name: str):
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        return nn.GELU()
