"""Tests for causal-LM datasets and collators."""

from __future__ import annotations

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import CausalLMCollator, LessonCausalLMDataset
from mopforge.tokenization import ByteTokenizer


def test_lesson_causal_lm_dataset_item_structure() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=512)

    item = dataset[0]

    assert set(item) == {
        "input_ids",
        "labels",
        "attention_mask",
        "lesson_id",
        "target_modules",
        "metadata",
        "domain",
        "skill",
    }
    assert item["lesson_id"] == lessons[0].id
    assert len(item["input_ids"]) == len(item["labels"])
    assert len(item["input_ids"]) == len(item["attention_mask"])
    assert item["input_ids"][0] == tokenizer.bos_token_id
    assert item["input_ids"][-1] == tokenizer.eos_token_id


def test_prompt_label_masking_uses_ignore_index() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=512)

    item = dataset[0]

    assert -100 in item["labels"]
    assert any(label != -100 for label in item["labels"])
    assert item["labels"][-1] == tokenizer.eos_token_id


def test_causal_lm_collator_padding_if_torch_installed() -> None:
    if CausalLMCollator is None:
        assert CausalLMCollator is None
        return

    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=256)
    collator = CausalLMCollator(tokenizer)

    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert str(batch["input_ids"].dtype) == "torch.int64"
    assert isinstance(batch["lesson_id"], list)
    assert isinstance(batch["target_modules"], list)
    assert isinstance(batch["metadata"], list)
