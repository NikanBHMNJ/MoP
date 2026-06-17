"""Optional tiny learned module router."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from mopforge.training.routing import DEFAULT_KNOWN_MODULES, normalize_target_modules


try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None
    TinyModuleRouter = None
else:

    class TinyModuleRouter(nn.Module):
        """Tiny multi-label router for predicting active MoP modules.

        This is a CPU smoke-test model: token embedding, masked mean pooling,
        and a small MLP that emits one logit per known module.
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 64,
            hidden_dim: int = 128,
            known_modules: Sequence[str] | None = None,
            pad_token_id: int = 0,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.known_modules = list(known_modules or DEFAULT_KNOWN_MODULES)
            if not self.known_modules:
                raise ValueError("known_modules must not be empty.")
            if len(self.known_modules) != len(set(self.known_modules)):
                raise ValueError("known_modules must not contain duplicates.")

            self.pad_token_id = pad_token_id
            self.embedding = nn.Embedding(
                vocab_size, d_model, padding_idx=pad_token_id
            )
            self.mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.known_modules)),
            )

        def forward(self, input_ids, attention_mask=None, module_mask=None):
            """Return router logits and optional BCE multi-label loss."""

            embedded = self.embedding(input_ids)
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long()

            mask = attention_mask.to(embedded.dtype).unsqueeze(-1)
            pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            logits = self.mlp(pooled)

            loss = None
            if module_mask is not None:
                labels = module_mask.to(logits.dtype)
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels)

            return {"logits": logits, "loss": loss}


def predict_modules(
    logits: Any,
    known_modules: Iterable[str],
    *,
    threshold: float = 0.5,
    always_include_core: bool = True,
) -> list[str] | list[list[str]]:
    """Convert router logits into normalized module names.

    A 1D logit vector returns ``list[str]``. A batch of logit vectors returns
    ``list[list[str]]``. Probabilities are computed with sigmoid.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    known_list = list(known_modules)
    rows = _logits_to_rows(logits)
    is_single = bool(rows and isinstance(rows[0], (int, float)))
    if is_single:
        return _predict_one(
            rows,
            known_list,
            threshold=threshold,
            always_include_core=always_include_core,
        )

    return [
        _predict_one(
            row,
            known_list,
            threshold=threshold,
            always_include_core=always_include_core,
        )
        for row in rows
    ]


def _predict_one(
    logits: Sequence[float],
    known_modules: list[str],
    *,
    threshold: float,
    always_include_core: bool,
) -> list[str]:
    if len(logits) != len(known_modules):
        raise ValueError("logits length must match known_modules length.")

    selected = [
        module
        for module, logit in zip(known_modules, logits)
        if _sigmoid(float(logit)) >= threshold
    ]
    return normalize_target_modules(
        selected,
        known_modules,
        always_include_core=always_include_core,
    )


def _logits_to_rows(logits: Any) -> Any:
    if hasattr(logits, "detach"):
        return logits.detach().cpu().tolist()
    return logits


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
