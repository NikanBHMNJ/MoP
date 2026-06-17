"""Tests for tiny fast adapters."""

from __future__ import annotations

import pytest

from mopforge.models import (
    FastAdapter,
    FastAdapterBank,
    FastAdapterConfig,
    TinyMoPCausalTransformer,
    adapter_names_from_target_modules,
    normalize_adapter_names,
)
from mopforge.training import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    build_optimizer_for_trainable_parameters,
    count_parameters,
)


def test_fast_adapter_config_validation() -> None:
    config = FastAdapterConfig(
        d_model=8,
        bottleneck_dim=4,
        adapter_names=["coding", "debugging"],
        residual_scale=0.5,
    )

    assert config.to_dict()["adapter_names"] == ["coding", "debugging"]
    with pytest.raises(ValueError, match="d_model"):
        FastAdapterConfig(d_model=0)
    with pytest.raises(ValueError, match="bottleneck"):
        FastAdapterConfig(d_model=8, bottleneck_dim=0)
    with pytest.raises(ValueError, match="unique"):
        FastAdapterConfig(d_model=8, adapter_names=["coding", "coding"])
    with pytest.raises(ValueError, match="residual_scale"):
        FastAdapterConfig(d_model=8, residual_scale=float("inf"))


def test_fast_adapter_preserves_input_shape() -> None:
    torch = pytest.importorskip("torch")
    if FastAdapter is None:
        pytest.skip("FastAdapter requires PyTorch.")
    adapter = FastAdapter(FastAdapterConfig(d_model=8, bottleneck_dim=4))
    hidden = torch.randn(2, 3, 8)

    output = adapter(hidden)

    assert output.shape == hidden.shape


def test_fast_adapter_bank_preserves_input_shape() -> None:
    torch = pytest.importorskip("torch")
    if FastAdapterBank is None:
        pytest.skip("FastAdapterBank requires PyTorch.")
    bank = FastAdapterBank(
        FastAdapterConfig(d_model=8, adapter_names=["coding", "debugging"])
    )
    hidden = torch.randn(2, 3, 8)

    output = bank(hidden, active_adapters=["coding"])

    assert output.shape == hidden.shape


def test_fast_adapter_bank_applies_named_adapter() -> None:
    torch = pytest.importorskip("torch")
    if FastAdapterBank is None:
        pytest.skip("FastAdapterBank requires PyTorch.")
    torch.manual_seed(123)
    bank = FastAdapterBank(FastAdapterConfig(d_model=8, adapter_names=["coding"]))
    hidden = torch.randn(2, 3, 8)

    output = bank(hidden, active_adapters="coding")

    assert not torch.allclose(output, hidden)


def test_fast_adapter_bank_multiple_active_adapters_are_deterministic() -> None:
    torch = pytest.importorskip("torch")
    if FastAdapterBank is None:
        pytest.skip("FastAdapterBank requires PyTorch.")
    torch.manual_seed(123)
    bank = FastAdapterBank(
        FastAdapterConfig(d_model=8, adapter_names=["coding", "debugging"])
    )
    bank.eval()
    hidden = torch.randn(2, 3, 8)

    first = bank(hidden, active_adapters=["coding", "debugging"])
    second = bank(hidden, active_adapters=["coding", "debugging"])

    assert torch.allclose(first, second)


def test_fast_adapter_bank_unknown_names_are_ignored() -> None:
    torch = pytest.importorskip("torch")
    if FastAdapterBank is None:
        pytest.skip("FastAdapterBank requires PyTorch.")
    bank = FastAdapterBank(FastAdapterConfig(d_model=8, adapter_names=["coding"]))
    hidden = torch.randn(2, 3, 8)

    output = bank(hidden, active_adapters=["unknown"])

    assert torch.equal(output, hidden)


def test_adapter_name_normalization_works() -> None:
    names = normalize_adapter_names(
        ["coding", "unknown", "coding", ""],
        known_adapters=["coding", "debugging"],
        include_default=True,
    )

    assert names == ["coding"]


def test_target_modules_map_to_adapter_names() -> None:
    assert adapter_names_from_target_modules(
        ["core", "coding", "debugging", "fast_adapter", "unknown"]
    ) == ["coding", "debugging", "default"]


def test_tiny_mop_forward_works_with_fast_adapters_disabled() -> None:
    torch = pytest.importorskip("torch")
    if TinyMoPCausalTransformer is None:
        pytest.skip("TinyMoPCausalTransformer requires PyTorch.")
    model = _tiny_mop(use_fast_adapters=False)
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)

    output = model(input_ids=input_ids, active_modules=["coding"])

    assert output["logits"].shape[:2] == input_ids.shape
    assert output["active_adapters"] == [[]]


def test_tiny_mop_forward_works_with_explicit_fast_adapters() -> None:
    torch = pytest.importorskip("torch")
    if TinyMoPCausalTransformer is None:
        pytest.skip("TinyMoPCausalTransformer requires PyTorch.")
    model = _tiny_mop(use_fast_adapters=True)
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)

    output = model(
        input_ids=input_ids,
        active_modules=["coding"],
        active_adapters=["coding"],
    )

    assert output["logits"].shape[:2] == input_ids.shape
    assert output["active_adapters"] == [["coding"]]


def test_fast_adapters_only_policy_trains_only_adapter_params() -> None:
    model = _tiny_mop(use_fast_adapters=True)

    summaries = apply_trainable_policy(
        model,
        TrainableParameterPolicy(mode="fast_adapters_only"),
    )
    by_name = {summary.name: summary for summary in summaries}

    assert by_name["adapter:coding"].trainable_params > 0
    assert by_name["adapter:debugging"].trainable_params > 0
    assert by_name["shared_core"].trainable_params == 0
    assert by_name["module:coding"].trainable_params == 0
    assert count_parameters(model)["trainable"] < count_parameters(model)["total"]


def test_optimizer_builder_sees_adapter_only_trainable_params() -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_mop(use_fast_adapters=True)
    apply_trainable_policy(model, TrainableParameterPolicy(mode="fast_adapters_only"))

    optimizer = build_optimizer_for_trainable_parameters(model, learning_rate=1e-3)
    optimizer_params = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]

    assert optimizer_params
    assert all(parameter.requires_grad for parameter in optimizer_params)
    assert sum(parameter.numel() for parameter in optimizer_params) == count_parameters(model)["trainable"]
    assert isinstance(optimizer, torch.optim.AdamW)


def test_fast_adapters_do_not_require_cuda() -> None:
    torch = pytest.importorskip("torch")

    assert torch.cuda.is_available() is False


def _tiny_mop(*, use_fast_adapters: bool):
    pytest.importorskip("torch")
    if TinyMoPCausalTransformer is None:
        pytest.skip("TinyMoPCausalTransformer requires PyTorch.")
    return TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
        module_names=["core", "coding", "debugging"],
        use_fast_adapters=use_fast_adapters,
        fast_adapter_names=["coding", "debugging", "repair"],
        fast_adapter_bottleneck_dim=4,
    )
