"""Tests for the optional tiny oracle-routed MoP model."""

from __future__ import annotations

import math

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import CausalLMCollator, LessonCausalLMDataset
from mopforge.models import TinyMoPCausalTransformer
from mopforge.tokenization import ByteTokenizer


def _build_tiny_batch():
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons[:2], tokenizer, max_length=512)
    return tokenizer, CausalLMCollator(tokenizer)([dataset[0], dataset[1]])


def test_tiny_mop_forward_shape_and_finite_loss_if_torch_installed() -> None:
    if CausalLMCollator is None or TinyMoPCausalTransformer is None:
        assert CausalLMCollator is None or TinyMoPCausalTransformer is None
        return

    tokenizer, batch = _build_tiny_batch()
    model = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=512,
        module_names=["core", "coding", "debugging", "math"],
    )

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        active_modules=batch["target_modules"],
    )

    assert outputs["logits"].shape[:2] == batch["input_ids"].shape
    assert outputs["logits"].shape[-1] == tokenizer.vocab_size
    assert outputs["loss"] is not None
    assert math.isfinite(outputs["loss"].item())
    assert outputs["active_modules"][0] == ["core", "coding", "debugging"]


def test_tiny_mop_different_active_modules_change_logits_if_torch_installed() -> None:
    if TinyMoPCausalTransformer is None:
        assert TinyMoPCausalTransformer is None
        return

    import torch

    torch.manual_seed(123)
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([[1, 68, 69, 70, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        module_names=["core", "coding", "debugging", "math"],
    )
    model.eval()

    with torch.no_grad():
        coding_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            active_modules=["coding"],
        )["logits"]
        math_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            active_modules=["math"],
        )["logits"]

    assert not torch.allclose(coding_logits, math_logits)
