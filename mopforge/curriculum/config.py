"""Configuration for deterministic curriculum planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CurriculumConfig:
    """Small deterministic curriculum scheduler settings."""

    strategy: str = "balanced"
    batch_size: int = 8
    max_lessons: int | None = None
    seed: int = 123
    domains: list[str] | None = None
    skills: list[str] | None = None
    target_modules: list[str] | None = None
    verification_statuses: list[str] | None = None
    difficulty_min: int | None = None
    difficulty_max: int | None = None
    include_repair: bool = True
    repair_boost: float = 1.0
    feedback_store_path: str | None = None
