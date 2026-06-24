"""Export production decoder weights to Hugging Face Llama-compatible artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from mopforge.models.architectures import ModelArchitectureConfig
from mopforge.tokenization import TokenizerSpec


def export_gpu_checkpoint_to_huggingface(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    expert_name: str | None = None,
    max_shard_size_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    """Restore a MoP-Forge GPU checkpoint and export a Llama-compatible model."""

    from mopforge.models.checkpoint_loader import load_gpu_checkpoint_model

    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        raise ValueError(
            "Hugging Face export currently requires a consolidated .pt checkpoint; "
            "a distributed sharded training checkpoint must be consolidated first."
        )
    loaded = load_gpu_checkpoint_model(checkpoint, config_path=config_path)
    payload = loaded["payload"]
    model = loaded["model"]
    architecture = loaded["architecture"]
    tokenizer_spec = loaded["tokenizer_spec"]
    restore_metadata = loaded["restore_metadata"]
    report = export_huggingface_llama(
        model,
        architecture,
        tokenizer_spec,
        output_dir,
        expert_name=expert_name,
        max_shard_size_bytes=max_shard_size_bytes,
    )
    report.update(
        {
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_format": payload.get("checkpoint_format"),
            "restore_metadata": restore_metadata,
        }
    )
    report_path = Path(output_dir) / "export_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def export_huggingface_llama(
    model,
    architecture: ModelArchitectureConfig,
    tokenizer_spec: TokenizerSpec,
    output_dir: str | Path,
    *,
    expert_name: str | None = None,
    max_shard_size_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    """Export Dense or one materialized MoP expert as ``LlamaForCausalLM``."""

    if architecture.architecture_family != "production_decoder_v2":
        raise ValueError("Hugging Face Llama export requires production_decoder_v2.")
    if architecture.model_type != "dense":
        names = list(architecture.module_names)
        if expert_name not in names:
            raise ValueError(
                f"MoP export requires expert_name from: {', '.join(names)}."
            )
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = _mapped_state_dict(model, architecture, expert_name=expert_name)
    shards = _write_shards(state, output, max_shard_size_bytes=max_shard_size_bytes)
    config = _hf_config(
        architecture,
        tokenizer_spec,
        expert_name=expert_name,
        torch_dtype=_torch_dtype_name(next(model.parameters()).dtype),
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "generation_config.json").write_text(
        json.dumps(
            {
                "_from_model_config": True,
                "bos_token_id": tokenizer_spec.bos_token_id,
                "eos_token_id": tokenizer_spec.eos_token_id,
                "pad_token_id": tokenizer_spec.pad_token_id,
                "transformers_version": "compatible",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tokenizer_files = _export_tokenizer(tokenizer_spec, output, architecture.max_seq_len)
    report = {
        "format": "mopforge_hf_llama_export_v1",
        "output_dir": str(output),
        "architecture_family": architecture.architecture_family,
        "source_model_type": architecture.model_type,
        "materialized_expert": expert_name,
        "weight_shards": shards,
        "tokenizer_files": tokenizer_files,
        "parameter_tensors": len(state),
        "total_weight_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in state.values()
        ),
    }
    (output / "export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _mapped_state_dict(model, architecture, *, expert_name):
    mapped = {
        "model.embed_tokens.weight": model.token_embedding.weight.detach().cpu(),
        "model.norm.weight": model.norm.weight.detach().cpu(),
    }
    if not architecture.tie_word_embeddings:
        mapped["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    layers = model.layers
    for index, block in enumerate(layers):
        source = f"model.layers.{index}"
        mapped.update(
            {
                f"{source}.input_layernorm.weight": block.input_norm.weight.detach().cpu(),
                f"{source}.post_attention_layernorm.weight": block.post_attention_norm.weight.detach().cpu(),
                f"{source}.self_attn.q_proj.weight": block.self_attn.q_proj.weight.detach().cpu(),
                f"{source}.self_attn.k_proj.weight": block.self_attn.k_proj.weight.detach().cpu(),
                f"{source}.self_attn.v_proj.weight": block.self_attn.v_proj.weight.detach().cpu(),
                f"{source}.self_attn.o_proj.weight": block.self_attn.o_proj.weight.detach().cpu(),
            }
        )
        mlp = block.mlp
        if architecture.model_type != "dense":
            mlp = mlp.experts[expert_name]
        mapped.update(
            {
                f"{source}.mlp.gate_proj.weight": mlp.gate_proj.weight.detach().cpu(),
                f"{source}.mlp.up_proj.weight": mlp.up_proj.weight.detach().cpu(),
                f"{source}.mlp.down_proj.weight": mlp.down_proj.weight.detach().cpu(),
            }
        )
    return mapped


def _write_shards(state, output: Path, *, max_shard_size_bytes: int):
    torch = _require_torch()
    partitions: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_bytes = 0
    for name, tensor in state.items():
        size = int(tensor.numel()) * int(tensor.element_size())
        if current and current_bytes + size > max_shard_size_bytes:
            partitions.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor.contiguous()
        current_bytes += size
    if current:
        partitions.append(current)
    count = len(partitions)
    weight_map = {}
    files = []
    for index, partition in enumerate(partitions, start=1):
        filename = (
            "pytorch_model.bin"
            if count == 1
            else f"pytorch_model-{index:05d}-of-{count:05d}.bin"
        )
        torch.save(partition, output / filename)
        files.append(filename)
        for name in partition:
            weight_map[name] = filename
    if count > 1:
        total_size = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in state.values()
        )
        (output / "pytorch_model.bin.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return files


def _hf_config(architecture, tokenizer_spec, *, expert_name, torch_dtype):
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": architecture.attention_dropout,
        "bos_token_id": tokenizer_spec.bos_token_id,
        "eos_token_id": tokenizer_spec.eos_token_id,
        "head_dim": architecture.d_model // architecture.n_heads,
        "hidden_act": "silu",
        "hidden_size": architecture.d_model,
        "initializer_range": 0.02,
        "intermediate_size": architecture.intermediate_size or architecture.d_model * 4,
        "max_position_embeddings": architecture.max_seq_len,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": architecture.n_heads,
        "num_hidden_layers": architecture.n_layers,
        "num_key_value_heads": architecture.n_key_value_heads or architecture.n_heads,
        "pad_token_id": tokenizer_spec.pad_token_id,
        "pretraining_tp": 1,
        "rms_norm_eps": architecture.rms_norm_eps,
        "rope_scaling": None,
        "rope_theta": architecture.rope_theta,
        "tie_word_embeddings": architecture.tie_word_embeddings,
        "torch_dtype": torch_dtype,
        "use_cache": True,
        "vocab_size": tokenizer_spec.vocab_size or architecture.vocab_size,
        "mopforge_source": {
            "model_type": architecture.model_type,
            "materialized_expert": expert_name,
        },
    }


def _export_tokenizer(spec: TokenizerSpec, output: Path, max_seq_len: int):
    files = []
    source = Path(spec.name_or_path or "")
    if source.is_file() and source.suffix.lower() == ".json":
        shutil.copy2(source, output / "tokenizer.json")
        files.append("tokenizer.json")
    special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    (output / "special_tokens_map.json").write_text(
        json.dumps(special_tokens, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "tokenizer_config.json").write_text(
        json.dumps(
            {
                **special_tokens,
                "clean_up_tokenization_spaces": False,
                "model_max_length": max_seq_len,
                "tokenizer_class": "PreTrainedTokenizerFast",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [*files, "special_tokens_map.json", "tokenizer_config.json"]


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Hugging Face export.") from exc
    return torch


def _torch_dtype_name(dtype) -> str:
    value = str(dtype).replace("torch.", "")
    aliases = {"float": "float32", "half": "float16"}
    return aliases.get(value, value)
