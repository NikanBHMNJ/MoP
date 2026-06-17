"""Tests for the optional tiny dense baseline model."""

from __future__ import annotations

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import CausalLMCollator, LessonCausalLMDataset
from mopforge.models import TinyCausalTransformer
from mopforge.tokenization import ByteTokenizer


def test_tiny_dense_model_forward_and_loss_if_torch_installed() -> None:
    if CausalLMCollator is None or TinyCausalTransformer is None:
        assert CausalLMCollator is None or TinyCausalTransformer is None
        return

    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons[:2], tokenizer, max_length=512)
    batch = CausalLMCollator(tokenizer)([dataset[0], dataset[1]])
    model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=512,
    )

    outputs = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    assert outputs["logits"].shape[:2] == batch["input_ids"].shape
    assert outputs["logits"].shape[-1] == tokenizer.vocab_size
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0
