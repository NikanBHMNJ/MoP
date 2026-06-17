"""Tests for oracle module routing helpers."""

from __future__ import annotations

import pytest

from mopforge.training import module_mask_from_targets, normalize_target_modules


def test_normalize_target_modules_is_stable_and_includes_core() -> None:
    known = ["core", "coding", "debugging", "math"]

    normalized = normalize_target_modules(
        ["debugging", "unknown", "coding", "coding"], known
    )

    assert normalized == ["core", "coding", "debugging"]


def test_normalize_target_modules_can_be_strict() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        normalize_target_modules(["coding", "router"], ["core", "coding"], strict=True)


def test_module_mask_from_targets_aligns_to_known_modules() -> None:
    known = ["core", "coding", "debugging", "math"]

    mask = module_mask_from_targets(["debugging"], known)

    assert mask == [1, 0, 1, 0]
