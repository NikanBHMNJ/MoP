"""Token-unit learning-rate scheduling for long pretraining runs."""

from __future__ import annotations

import math


class TokenLRScheduler:
    def __init__(
        self,
        optimizer,
        *,
        scheduler: str,
        total_tokens: int,
        warmup_tokens: int = 0,
        min_lr_ratio: float = 0.0,
    ) -> None:
        if scheduler not in {"cosine", "linear_warmup"}:
            raise ValueError("TokenLRScheduler supports cosine or linear_warmup.")
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive.")
        if warmup_tokens < 0 or warmup_tokens >= total_tokens:
            raise ValueError("warmup_tokens must be in [0, total_tokens).")
        if not 0.0 <= float(min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1].")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_tokens = int(total_tokens)
        self.warmup_tokens = int(warmup_tokens)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.tokens_seen = 0

    def step(self, tokens_seen: int | None = None) -> None:
        self.tokens_seen = int(self.tokens_seen + 1 if tokens_seen is None else tokens_seen)
        ratio = self._ratio(self.tokens_seen)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * ratio

    def state_dict(self) -> dict:
        return {
            "scheduler": self.scheduler,
            "total_tokens": self.total_tokens,
            "warmup_tokens": self.warmup_tokens,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state: dict) -> None:
        self.tokens_seen = int(state.get("tokens_seen", 0))
        saved = list(state.get("base_lrs") or self.base_lrs)
        if len(saved) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler optimizer group count changed across resume.")
        self.base_lrs = [float(value) for value in saved]
        self.step(self.tokens_seen)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _ratio(self, tokens_seen: int) -> float:
        if self.warmup_tokens and tokens_seen < self.warmup_tokens:
            return max(1e-12, tokens_seen / float(self.warmup_tokens))
        if self.scheduler == "linear_warmup":
            return 1.0
        progress = min(
            1.0,
            max(0.0, tokens_seen - self.warmup_tokens)
            / float(max(1, self.total_tokens - self.warmup_tokens)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
