import pytest

from mopforge.generation import generate_greedy
from mopforge.models import (
    ModelArchitectureConfig,
    ProductionCausalLM,
    ProductionDecoderConfig,
    build_tiny_model_from_architecture,
    production_parameter_count,
)
from mopforge.tokenization import ByteTokenizer
from mopforge.training.parameter_policy import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    count_parameters,
)


torch = pytest.importorskip("torch")


def _config(model_type="dense"):
    return ProductionDecoderConfig(
        vocab_size=67,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        model_type=model_type,
        module_names=("coding", "debugging"),
        active_experts=1,
        dropout=0.0,
        attention_dropout=0.0,
    )


def test_production_dense_forward_loss_and_analytic_parameter_count():
    config = _config()
    model = ProductionCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    outputs = model(input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert torch.isfinite(outputs["loss"])
    assert count_parameters(model)["total"] == production_parameter_count(config)


def test_production_native_kv_cache_matches_full_decode():
    config = _config()
    model = ProductionCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 6))
    next_token = torch.randint(0, config.vocab_size, (1, 1))

    with torch.no_grad():
        prefill = model(prompt, use_cache=True)
        decoded = model(
            next_token,
            past_key_values=prefill["past_key_values"],
            use_cache=True,
        )
        complete = model(torch.cat((prompt, next_token), dim=1))

    assert len(prefill["past_key_values"]) == config.n_layers
    assert prefill["past_key_values"][0][0].shape[1] == config.n_key_value_heads
    assert torch.allclose(
        decoded["logits"][:, -1],
        complete["logits"][:, -1],
        atol=1e-5,
        rtol=1e-4,
    )


def test_production_mop_routes_experts_and_supports_sparse_tail():
    config = _config("mop_oracle")
    model = ProductionCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 7))

    with torch.no_grad():
        encoded = model.encode_for_sparse_tail(
            input_ids,
            active_modules=["coding"],
        )
        tail = model.forward_from_hidden(
            encoded["hidden_states"],
            active_modules=["coding"],
        )
        full = model(input_ids, active_modules=["coding"])

    assert torch.allclose(tail["logits"], full["logits"], atol=1e-6)
    assert model.last_forward_metadata["routed_block_metadata"]

    apply_trainable_policy(
        model,
        TrainableParameterPolicy(mode="target_modules_only", target_modules=["coding"]),
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("experts.coding" in name for name in trainable)


def test_architecture_builder_and_generation_support_production_decoder():
    tokenizer = ByteTokenizer()
    architecture = ModelArchitectureConfig(
        name="production-smoke",
        architecture_family="production_decoder_v2",
        model_type="dense",
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=32,
    )
    model = build_tiny_model_from_architecture(architecture, tokenizer=tokenizer)

    generated = generate_greedy(model, tokenizer, "abc", max_new_tokens=2)

    assert isinstance(model, ProductionCausalLM)
    assert isinstance(generated, str)
