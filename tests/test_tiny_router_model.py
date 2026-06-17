"""Tests for the tiny learned module router."""

from __future__ import annotations

import math

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import RouterCollator, RouterDataset
from mopforge.models import TinyModuleRouter, TinyMoPCausalTransformer, predict_modules
from mopforge.tokenization import ByteTokenizer
from mopforge.training import DEFAULT_KNOWN_MODULES, route_batch_with_router


def _router_batch():
    lessons = generate_coding_bugfix_lessons(count_per_category=1, verify=False)[:2]
    tokenizer = ByteTokenizer()
    dataset = RouterDataset(
        lessons,
        tokenizer,
        known_modules=DEFAULT_KNOWN_MODULES,
        max_length=256,
    )
    return tokenizer, RouterCollator(tokenizer)([dataset[0], dataset[1]])


def test_tiny_router_forward_shape_and_finite_loss_if_torch_installed() -> None:
    if RouterCollator is None or TinyModuleRouter is None:
        assert RouterCollator is None or TinyModuleRouter is None
        return

    tokenizer, batch = _router_batch()
    router = TinyModuleRouter(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        hidden_dim=64,
        known_modules=DEFAULT_KNOWN_MODULES,
        pad_token_id=tokenizer.pad_token_id,
    )

    outputs = router(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        module_mask=batch["module_mask"],
    )

    assert outputs["logits"].shape == (2, len(DEFAULT_KNOWN_MODULES))
    assert outputs["loss"] is not None
    assert math.isfinite(outputs["loss"].item())


def test_predict_modules_always_includes_core() -> None:
    logits = [-10.0] * len(DEFAULT_KNOWN_MODULES)

    predicted = predict_modules(
        logits,
        DEFAULT_KNOWN_MODULES,
        threshold=0.99,
        always_include_core=True,
    )

    assert predicted == ["core"]


def test_tiny_router_one_optimization_step_is_finite_if_torch_installed() -> None:
    if RouterCollator is None or TinyModuleRouter is None:
        assert RouterCollator is None or TinyModuleRouter is None
        return

    import torch

    tokenizer, batch = _router_batch()
    router = TinyModuleRouter(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        hidden_dim=64,
        known_modules=DEFAULT_KNOWN_MODULES,
        pad_token_id=tokenizer.pad_token_id,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    loss = router(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        module_mask=batch["module_mask"],
    )["loss"]
    loss.backward()
    optimizer.step()

    assert math.isfinite(loss.item())


def test_route_batch_with_router_outputs_modules_if_torch_installed() -> None:
    if (
        RouterCollator is None
        or TinyModuleRouter is None
        or TinyMoPCausalTransformer is None
    ):
        assert RouterCollator is None or TinyModuleRouter is None
        return

    tokenizer, batch = _router_batch()
    router = TinyModuleRouter(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        hidden_dim=64,
        known_modules=DEFAULT_KNOWN_MODULES,
        pad_token_id=tokenizer.pad_token_id,
    )

    predicted = route_batch_with_router(
        router,
        batch,
        DEFAULT_KNOWN_MODULES,
        threshold=0.5,
    )

    assert len(predicted) == 2
    assert all("core" in modules for modules in predicted)
