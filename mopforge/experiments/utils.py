"""Utility helpers for tiny MoP-Forge experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def set_seed(seed: int) -> None:
    """Set Python and optional PyTorch random seeds."""

    random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)


def split_lessons(
    lessons: Sequence[T],
    train_fraction: float = 0.8,
    seed: int = 123,
) -> tuple[list[T], list[T]]:
    """Split lessons deterministically into train and eval lists."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    shuffled = list(lessons)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    return shuffled[:split_index], shuffled[split_index:]


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or NaN for an empty sequence."""

    if not values:
        return float("nan")
    return sum(values) / len(values)
