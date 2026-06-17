"""Tests for learned-router datasets and collators."""

from __future__ import annotations

import pytest

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import RouterCollator, RouterDataset
from mopforge.tokenization import ByteTokenizer
from mopforge.training import DEFAULT_KNOWN_MODULES


def test_router_dataset_item_includes_module_mask() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]
    tokenizer = ByteTokenizer()
    dataset = RouterDataset([lesson], tokenizer, known_modules=DEFAULT_KNOWN_MODULES)

    item = dataset[0]

    assert set(item) == {
        "input_ids",
        "attention_mask",
        "module_mask",
        "target_modules",
        "metadata",
        "lesson_id",
        "route_text",
    }
    assert item["module_mask"] == [1, 1, 1, 0, 0, 0, 0]
    assert "Expected" not in item["route_text"]
    assert lesson.expected_output.rstrip() not in item["route_text"]


def test_router_dataset_unknown_modules_follow_routing_strictness() -> None:
    lesson = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[0]
    lesson.target_modules.append("unknown_module")

    relaxed = RouterDataset([lesson], known_modules=DEFAULT_KNOWN_MODULES)
    assert relaxed[0]["module_mask"] == [1, 1, 1, 0, 0, 0, 0]

    strict = RouterDataset(
        [lesson],
        known_modules=DEFAULT_KNOWN_MODULES,
        strict_modules=True,
    )
    with pytest.raises(ValueError, match="Unknown"):
        strict[0]


def test_router_collator_pads_and_stacks_if_torch_installed() -> None:
    if RouterCollator is None:
        assert RouterCollator is None
        return

    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[:2]
    tokenizer = ByteTokenizer()
    dataset = RouterDataset(lessons, tokenizer, known_modules=DEFAULT_KNOWN_MODULES)
    first = dataset[0]
    second = dataset[1]

    batch = RouterCollator(tokenizer)([first, second])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["module_mask"].shape == (2, len(DEFAULT_KNOWN_MODULES))
    assert str(batch["input_ids"].dtype) == "torch.int64"
    assert str(batch["module_mask"].dtype) == "torch.float32"
    assert batch["input_ids"].shape[1] == max(
        len(first["input_ids"]), len(second["input_ids"])
    )
    assert isinstance(batch["target_modules"], list)
    assert isinstance(batch["metadata"], list)
