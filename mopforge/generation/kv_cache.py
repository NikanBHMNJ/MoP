"""Incremental K/V-cached decoding for MoP-Forge transformer smoke models."""

from __future__ import annotations

from typing import Any


def supports_kv_cache(model: Any) -> tuple[bool, str | None]:
    """Return whether the model layout has an exact incremental implementation."""

    layers = _transformer_layers(model)
    if layers is None:
        return False, "model does not expose supported transformer encoder layers"
    if getattr(model, "routed_blocks", None):
        return False, "routed_ffn incremental decoding is not implemented"
    if any(hasattr(layer, "q_lora_bank") for layer in layers):
        return False, "internal attention LoRA incremental decoding is not implemented"
    if any(bool(getattr(layer, "norm_first", False)) for layer in layers):
        return False, "pre-norm transformer incremental decoding is not implemented"
    return True, None


def kv_cache_prefill(
    model: Any,
    input_ids,
    *,
    active_modules=None,
    active_adapters=None,
    active_conditions=None,
) -> tuple[Any, list[tuple[Any, Any]], dict[str, Any]]:
    """Prefill a causal prompt and return logits plus per-layer K/V tensors."""

    torch = _require_torch()
    supported, reason = supports_kv_cache(model)
    if not supported:
        raise ValueError(f"K/V cache is unavailable: {reason}.")
    batch_size, seq_len = input_ids.shape
    if seq_len > int(model.max_seq_len):
        raise ValueError("Prompt exceeds model.max_seq_len.")
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
    hidden = model.dropout(hidden)
    cache: list[tuple[Any, Any]] = []
    for layer in _transformer_layers(model):
        hidden, key, value = _encoder_layer_prefill(layer, hidden)
        cache.append((key, value))
    logits, metadata = _tail_logits(
        model,
        hidden,
        active_modules=active_modules,
        active_adapters=active_adapters,
        active_conditions=active_conditions,
    )
    metadata.update(
        {
            "kv_cache_enabled": True,
            "kv_cache_layers": len(cache),
            "kv_cache_tokens": seq_len,
        }
    )
    return logits, cache, metadata


def kv_cache_decode_token(
    model: Any,
    input_ids,
    cache: list[tuple[Any, Any]],
    *,
    position: int,
    active_modules=None,
    active_adapters=None,
    active_conditions=None,
) -> tuple[Any, list[tuple[Any, Any]], dict[str, Any]]:
    """Decode one token and append its projected keys/values to the cache."""

    torch = _require_torch()
    if input_ids.shape[1] != 1:
        raise ValueError("kv_cache_decode_token expects exactly one token.")
    if position < 0 or position >= int(model.max_seq_len):
        raise ValueError("K/V cache position exceeds model.max_seq_len.")
    layers = _transformer_layers(model)
    if layers is None or len(layers) != len(cache):
        raise ValueError("K/V cache layer count does not match the model.")
    positions = torch.tensor([[position]], dtype=torch.long, device=input_ids.device)
    hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
    hidden = model.dropout(hidden)
    next_cache: list[tuple[Any, Any]] = []
    for layer, (past_key, past_value) in zip(layers, cache):
        hidden, key, value = _encoder_layer_decode(
            layer,
            hidden,
            past_key,
            past_value,
        )
        next_cache.append((key, value))
    logits, metadata = _tail_logits(
        model,
        hidden,
        active_modules=active_modules,
        active_adapters=active_adapters,
        active_conditions=active_conditions,
    )
    metadata.update(
        {
            "kv_cache_enabled": True,
            "kv_cache_layers": len(next_cache),
            "kv_cache_tokens": position + 1,
        }
    )
    return logits, next_cache, metadata


def _encoder_layer_prefill(layer, hidden):
    torch = _require_torch()
    functional = torch.nn.functional
    query, key, value = _project_qkv(layer.self_attn, hidden)
    attention = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )
    attention = _merge_heads(attention)
    attention = layer.self_attn.out_proj(attention)
    hidden = layer.norm1(hidden + layer.dropout1(attention))
    feed_forward = layer.linear2(
        layer.dropout(layer.activation(layer.linear1(hidden)))
    )
    hidden = layer.norm2(hidden + layer.dropout2(feed_forward))
    return hidden, key, value


def _encoder_layer_decode(layer, hidden, past_key, past_value):
    torch = _require_torch()
    functional = torch.nn.functional
    query, current_key, current_value = _project_qkv(layer.self_attn, hidden)
    key = torch.cat((past_key, current_key), dim=2)
    value = torch.cat((past_value, current_value), dim=2)
    attention = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )
    attention = _merge_heads(attention)
    attention = layer.self_attn.out_proj(attention)
    hidden = layer.norm1(hidden + layer.dropout1(attention))
    feed_forward = layer.linear2(
        layer.dropout(layer.activation(layer.linear1(hidden)))
    )
    hidden = layer.norm2(hidden + layer.dropout2(feed_forward))
    return hidden, key, value


def _project_qkv(attention, hidden):
    torch = _require_torch()
    functional = torch.nn.functional
    weight = attention.in_proj_weight
    bias = attention.in_proj_bias
    query, key, value = functional.linear(hidden, weight, bias).chunk(3, dim=-1)
    batch_size, seq_len, embed_dim = query.shape
    heads = int(attention.num_heads)
    head_dim = embed_dim // heads

    def split_heads(tensor):
        return tensor.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)

    return split_heads(query), split_heads(key), split_heads(value)


def _merge_heads(hidden):
    batch_size, heads, seq_len, head_dim = hidden.shape
    return hidden.transpose(1, 2).contiguous().view(
        batch_size,
        seq_len,
        heads * head_dim,
    )


def _tail_logits(
    model,
    hidden,
    *,
    active_modules,
    active_adapters,
    active_conditions,
):
    batch_size = hidden.shape[0]
    if hasattr(model, "forward_from_hidden"):
        modules = model._expand_active_modules(active_modules, batch_size)
        if getattr(model, "module_bank", None) is not None:
            hidden = model._apply_module_bank(hidden, modules)
        outputs = model.forward_from_hidden(
            hidden,
            active_modules=modules,
            active_adapters=active_adapters,
            active_conditions=active_conditions,
        )
        return outputs["logits"], {
            "model_path": "mop_post_core",
            "active_modules": modules,
        }

    adapters = model._expand_active_adapters(active_adapters, batch_size)
    if model.fast_adapter_bank is not None:
        hidden = model.fast_adapter_bank(hidden, active_adapters=adapters)
    conditions = model._expand_active_conditions(active_conditions, batch_size)
    if model.generated_adapter is not None:
        hidden = model.generated_adapter(hidden, active_conditions=conditions)
    logits = model.lm_head(model.norm(hidden))
    return logits, {
        "model_path": "dense",
        "active_adapters": adapters,
        "active_conditions": conditions,
    }


def _transformer_layers(model):
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        blocks = getattr(model, "shared_blocks", None)
    layers = getattr(blocks, "layers", None)
    return list(layers) if layers is not None else None


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for K/V-cached generation.") from exc
    return torch
