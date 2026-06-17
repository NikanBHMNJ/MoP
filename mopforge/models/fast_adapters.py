"""Optional tiny named fast adapters for CPU smoke experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


DEFAULT_ADAPTER_MODULE_MAP = {
    "coding": "coding",
    "debugging": "debugging",
    "repair": "repair",
    "router": "router",
    "fast_adapter": "default",
}


@dataclass(slots=True)
class FastAdapterConfig:
    """Configuration for a tiny named adapter bank."""

    d_model: int
    bottleneck_dim: int = 16
    adapter_names: list[str] = field(default_factory=lambda: ["default"])
    dropout: float = 0.0
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate adapter dimensions and names."""

        if type(self.d_model) is not int or self.d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        if type(self.bottleneck_dim) is not int or self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be a positive integer.")
        if not isinstance(self.adapter_names, list):
            self.adapter_names = list(self.adapter_names)
        if not self.adapter_names or not all(
            isinstance(name, str) and name.strip()
            for name in self.adapter_names
        ):
            raise ValueError("adapter_names must contain non-empty strings.")
        seen = set()
        normalized = []
        for name in self.adapter_names:
            if name in seen:
                raise ValueError("adapter_names must be unique.")
            seen.add(name)
            normalized.append(name)
        self.adapter_names = normalized
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0.0, 1.0).")
        if not math.isfinite(float(self.residual_scale)):
            raise ValueError("residual_scale must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return {
            "d_model": self.d_model,
            "bottleneck_dim": self.bottleneck_dim,
            "adapter_names": list(self.adapter_names),
            "dropout": float(self.dropout),
            "residual_scale": float(self.residual_scale),
        }


def normalize_adapter_names(
    names,
    known_adapters: Iterable[str] | None = None,
    *,
    include_default: bool = False,
) -> list[str]:
    """Normalize adapter names, ignoring unknown names when a known set is given."""

    known_list = list(known_adapters) if known_adapters is not None else None
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


def adapter_names_from_target_modules(target_modules: list[str] | str | None) -> list[str]:
    """Map lesson target modules to known adapter names."""

    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    mapped = [
        DEFAULT_ADAPTER_MODULE_MAP[module]
        for module in target_modules
        if module in DEFAULT_ADAPTER_MODULE_MAP
    ]
    return normalize_adapter_names(mapped)


try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None
    FastAdapter = None
    FastAdapterBank = None
else:

    class FastAdapter(nn.Module):
        """Tiny bottleneck adapter that returns ``x + residual_scale * delta``."""

        def __init__(self, config: FastAdapterConfig) -> None:
            super().__init__()
            self.config = config
            self.norm = nn.LayerNorm(config.d_model)
            self.down = nn.Linear(config.d_model, config.bottleneck_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(config.dropout)
            self.up = nn.Linear(config.bottleneck_dim, config.d_model)

        def forward(self, hidden_states):
            """Apply the adapter residual update."""

            delta = self.up(
                self.dropout(self.activation(self.down(self.norm(hidden_states))))
            )
            return hidden_states + float(self.config.residual_scale) * delta


    class FastAdapterBank(nn.Module):
        """A named collection of tiny fast adapters.

        ``active_adapters=None`` applies no adapter. Unknown adapter names are
        ignored. Multiple active adapters are combined by averaging their deltas.
        Batch-style active adapters are supported as a list with one name/list
        per batch row.
        """

        def __init__(self, config: FastAdapterConfig) -> None:
            super().__init__()
            self.config = config
            self.adapter_names = list(config.adapter_names)
            self.adapters = nn.ModuleDict(
                {
                    adapter_name: FastAdapter(config)
                    for adapter_name in self.adapter_names
                }
            )

        def forward(self, hidden_states, active_adapters=None):
            """Apply selected adapters to hidden states."""

            if active_adapters is None:
                return hidden_states
            if isinstance(active_adapters, str):
                names = normalize_adapter_names(active_adapters, self.adapter_names)
                return self._apply_names(hidden_states, names)

            active_list = list(active_adapters)
            if not active_list:
                return hidden_states
            if all(isinstance(item, str) for item in active_list):
                names = normalize_adapter_names(active_list, self.adapter_names)
                return self._apply_names(hidden_states, names)
            if len(active_list) != hidden_states.shape[0]:
                raise ValueError("Batch-style active_adapters length must match batch size.")

            chunks = []
            for index, adapter_names in enumerate(active_list):
                names = normalize_adapter_names(adapter_names, self.adapter_names)
                chunks.append(
                    self._apply_names(hidden_states[index : index + 1], names)
                )
            return torch.cat(chunks, dim=0)

        def _apply_names(self, hidden_states, names: Sequence[str]):
            if not names:
                return hidden_states
            deltas = [
                self.adapters[name](hidden_states) - hidden_states
                for name in names
                if name in self.adapters
            ]
            if not deltas:
                return hidden_states
            delta = torch.stack(deltas, dim=0).mean(dim=0)
            return hidden_states + delta
