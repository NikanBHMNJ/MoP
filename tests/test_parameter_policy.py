"""Tests for trainable-parameter policies and parameter grouping."""

from __future__ import annotations

import pytest

from mopforge.queues import TrainingQueueItem
from mopforge.training import (
    DEFAULT_KNOWN_MODULES,
    TrainableParameterPolicy,
    apply_trainable_policy,
    build_optimizer_for_trainable_parameters,
    count_parameters,
    infer_parameter_group,
    policy_from_queue_item,
)


def test_trainable_parameter_policy_validation() -> None:
    with pytest.raises(ValueError, match="mode"):
        TrainableParameterPolicy(mode="sometimes")
    with pytest.raises(ValueError, match="target_modules"):
        TrainableParameterPolicy(mode="target_modules_only", target_modules=["coding", ""])

    policy = TrainableParameterPolicy(
        mode="target_modules_only",
        target_modules=["coding", "coding", "debugging"],
    )

    assert policy.target_modules == ["coding", "debugging"]


def test_infer_parameter_group_stable_categories() -> None:
    assert infer_parameter_group("token_embedding.weight") == "embeddings"
    assert infer_parameter_group("position_embedding.weight") == "embeddings"
    assert infer_parameter_group("shared_blocks.layers.0.linear1.weight") == "shared_core"
    assert infer_parameter_group("blocks.layers.0.linear1.weight") == "shared_core"
    assert infer_parameter_group("module_bank.blocks.coding.net.1.weight") == "module:coding"
    assert infer_parameter_group("module_bank.blocks.debugging.net.1.bias") == "module:debugging"
    assert infer_parameter_group("router.mlp.1.weight") == "router"
    assert infer_parameter_group("lm_head.weight") == "lm_head"
    assert infer_parameter_group("norm.weight") == "norm"
    assert infer_parameter_group("misc.weight") == "other"


def test_count_parameters_returns_total_trainable_and_frozen() -> None:
    model = _tiny_dense()

    counts = count_parameters(model)

    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]
    assert counts["frozen"] == 0


def test_apply_trainable_policy_all_leaves_everything_trainable() -> None:
    model = _tiny_dense()
    apply_trainable_policy(model, TrainableParameterPolicy(mode="frozen"))

    summaries = apply_trainable_policy(model, TrainableParameterPolicy(mode="all"))

    assert count_parameters(model)["frozen"] == 0
    assert sum(summary.trainable_params for summary in summaries) == count_parameters(model)["total"]


def test_apply_trainable_policy_frozen_freezes_everything() -> None:
    model = _tiny_dense()

    summaries = apply_trainable_policy(model, TrainableParameterPolicy(mode="frozen"))

    assert count_parameters(model)["trainable"] == 0
    assert all(summary.frozen_params == summary.total_params for summary in summaries)


def test_core_only_freezes_tiny_mop_module_bank() -> None:
    model = _tiny_mop()

    summaries = apply_trainable_policy(model, TrainableParameterPolicy(mode="core_only"))
    by_name = {summary.name: summary for summary in summaries}

    module_summaries = [
        summary for summary in summaries if summary.name.startswith("module:")
    ]
    assert module_summaries
    assert all(summary.trainable_params == 0 for summary in module_summaries)
    assert by_name["shared_core"].trainable_params > 0


def test_modules_only_freezes_shared_core_and_trains_module_bank() -> None:
    model = _tiny_mop()

    summaries = apply_trainable_policy(model, TrainableParameterPolicy(mode="modules_only"))
    by_name = {summary.name: summary for summary in summaries}
    module_summaries = [
        summary for summary in summaries if summary.name.startswith("module:")
    ]

    assert by_name["shared_core"].trainable_params == 0
    assert module_summaries
    assert all(summary.trainable_params > 0 for summary in module_summaries)


def test_target_modules_only_selects_named_tiny_mop_module() -> None:
    model = _tiny_mop()

    summaries = apply_trainable_policy(
        model,
        TrainableParameterPolicy(
            mode="target_modules_only",
            target_modules=["coding"],
        ),
    )
    by_name = {summary.name: summary for summary in summaries}

    assert by_name["module:coding"].trainable_params > 0
    assert by_name["module:debugging"].trainable_params == 0
    assert by_name["shared_core"].trainable_params == 0


def test_optimizer_builder_only_includes_trainable_parameters() -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_dense()
    apply_trainable_policy(model, TrainableParameterPolicy(mode="head_only"))

    optimizer = build_optimizer_for_trainable_parameters(model, learning_rate=1e-3)

    optimizer_params = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert optimizer_params
    assert all(parameter.requires_grad for parameter in optimizer_params)
    assert sum(parameter.numel() for parameter in optimizer_params) == count_parameters(model)["trainable"]
    assert isinstance(optimizer, torch.optim.AdamW)


def test_policy_from_queue_item_creates_target_module_policy() -> None:
    item = TrainingQueueItem(
        item_id="queue-debugging-lesson-a",
        module="debugging",
        lesson_id="lesson-a",
    )

    policy = policy_from_queue_item(item)

    assert policy.mode == "target_modules_only"
    assert policy.target_modules == ["debugging"]
    assert policy.metadata["queue_item_id"] == "queue-debugging-lesson-a"
    assert policy.metadata["lesson_id"] == "lesson-a"


def test_policy_from_queue_item_can_create_fast_adapter_policy() -> None:
    item = TrainingQueueItem(
        item_id="queue-fast-adapter-lesson-a",
        module="fast_adapter",
        lesson_id="lesson-a",
        metadata={"adapter_name": "coding"},
    )

    policy = policy_from_queue_item(item)

    assert policy.mode == "fast_adapters_only"
    assert policy.target_modules == ["coding"]
    assert policy.train_fast_adapters is True
    assert policy.metadata["module"] == "fast_adapter"


def test_policy_helpers_do_not_require_cuda() -> None:
    torch = pytest.importorskip("torch")

    assert torch.cuda.is_available() is False


def _tiny_dense():
    pytest.importorskip("torch")
    from mopforge.models import TinyCausalTransformer

    if TinyCausalTransformer is None:
        pytest.skip("TinyCausalTransformer requires PyTorch.")
    return TinyCausalTransformer(
        vocab_size=64,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
    )


def _tiny_mop():
    pytest.importorskip("torch")
    from mopforge.models import TinyMoPCausalTransformer

    if TinyMoPCausalTransformer is None:
        pytest.skip("TinyMoPCausalTransformer requires PyTorch.")
    return TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
        module_names=DEFAULT_KNOWN_MODULES,
    )
