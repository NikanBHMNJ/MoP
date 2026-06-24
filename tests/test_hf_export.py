import json

import pytest

from mopforge.models import (
    ModelArchitectureConfig,
    build_tiny_model_from_architecture,
    export_gpu_checkpoint_to_huggingface,
    export_huggingface_llama,
)
from mopforge.gpu import GPUTrainingConfig, GPUTrainingState, save_gpu_checkpoint
from mopforge.tokenization import (
    BPETrainingConfig,
    TokenizerSpec,
    build_tokenizer,
    train_bpe_tokenizer,
)


torch = pytest.importorskip("torch")


def _architecture(model_type="dense"):
    return ModelArchitectureConfig(
        name="hf-export-smoke",
        architecture_family="production_decoder_v2",
        model_type=model_type,
        vocab_size=67,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        module_names=["coding", "debugging"],
        active_experts=1,
    )


def _tokenizer_spec(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    return TokenizerSpec(
        tokenizer_type="hf",
        name_or_path=str(tokenizer_path),
        vocab_size=67,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
    )


def test_dense_hf_export_writes_llama_config_and_sharded_index(tmp_path):
    architecture = _architecture()
    model = build_tiny_model_from_architecture(architecture)
    report = export_huggingface_llama(
        model,
        architecture,
        _tokenizer_spec(tmp_path),
        tmp_path / "export",
        max_shard_size_bytes=3000,
    )

    config = json.loads((tmp_path / "export" / "config.json").read_text())
    index = json.loads(
        (tmp_path / "export" / "pytorch_model.bin.index.json").read_text()
    )
    assert config["model_type"] == "llama"
    assert config["num_key_value_heads"] == 2
    assert len(report["weight_shards"]) > 1
    assert "model.embed_tokens.weight" in index["weight_map"]
    assert not any(name.startswith("router") for name in index["weight_map"])

    transformers = pytest.importorskip("transformers")
    exported = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "export",
        local_files_only=True,
    )
    model.eval()
    exported.eval()
    input_ids = torch.tensor([[1, 8, 12, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        observed = exported(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)


def test_mop_hf_export_requires_and_materializes_named_expert(tmp_path):
    architecture = _architecture("mop_oracle")
    model = build_tiny_model_from_architecture(architecture)
    with pytest.raises(ValueError):
        export_huggingface_llama(
            model,
            architecture,
            _tokenizer_spec(tmp_path),
            tmp_path / "missing-expert",
        )

    report = export_huggingface_llama(
        model,
        architecture,
        _tokenizer_spec(tmp_path),
        tmp_path / "coding-expert",
        expert_name="coding",
    )

    assert report["materialized_expert"] == "coding"
    state = torch.load(
        tmp_path / "coding-expert" / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(
        state["model.layers.0.mlp.gate_proj.weight"],
        model.layers[0].mlp.experts["coding"].gate_proj.weight,
    )


def test_gpu_checkpoint_export_restores_model_and_local_bpe(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("def add(a, b): return a + b\n" * 20, encoding="utf-8")
    tokenizer_report = train_bpe_tokenizer(
        BPETrainingConfig(
            source_paths=[str(corpus)],
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=280,
            min_frequency=1,
        )
    )
    tokenizer_spec = TokenizerSpec.load_json(tokenizer_report["tokenizer_spec_path"])
    tokenizer = build_tokenizer(tokenizer_spec)
    architecture = ModelArchitectureConfig(
        name="checkpoint-export",
        architecture_family="production_decoder_v2",
        model_type="dense",
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=32,
    )
    model = build_tiny_model_from_architecture(architecture, tokenizer=tokenizer)
    config = GPUTrainingConfig(
        name="checkpoint-export",
        architecture_family="production_decoder_v2",
        tokenizer_type="hf",
        tokenizer_spec_path=tokenizer_report["tokenizer_spec_path"],
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        device="cpu",
        precision="fp32",
        enable_amp=False,
    )
    checkpoint = tmp_path / "model.pt"
    save_gpu_checkpoint(
        checkpoint,
        model=model,
        state=GPUTrainingState(),
        config=config,
        model_metadata={"architecture": architecture.to_dict()},
    )

    report = export_gpu_checkpoint_to_huggingface(
        checkpoint,
        tmp_path / "hf",
        max_shard_size_bytes=10_000_000,
    )

    assert report["source_checkpoint"] == str(checkpoint)
    assert (tmp_path / "hf" / "tokenizer.json").is_file()
    restored = torch.load(
        tmp_path / "hf" / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(
        restored["model.layers.0.self_attn.q_proj.weight"],
        model.layers[0].self_attn.q_proj.weight,
    )
