"""Configuration for tiny CPU-safe comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from mopforge.training import DEFAULT_KNOWN_MODULES


@dataclass(slots=True)
class TinyExperimentConfig:
    """Small CPU-safe settings for dense-vs-MoP smoke comparisons."""

    seed: int = 123
    lesson_path: str = "data/coding_bugfix_lessons.jsonl"
    batch_size: int = 2
    train_steps: int = 3
    eval_batches: int = 2
    max_seq_len: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    router_train_steps: int = 3
    router_hidden_dim: int = 128
    learning_rate: float = 1e-3
    train_fraction: float = 0.8
    run_generation_eval: bool = False
    generation_eval_examples: int = 3
    max_new_tokens: int = 128
    known_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_KNOWN_MODULES)
    )
