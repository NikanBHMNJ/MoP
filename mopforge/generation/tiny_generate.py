"""Greedy generation for tiny MoP-Forge causal-LM smoke models."""

from __future__ import annotations

from typing import Any

from mopforge.tokenization import (
    get_tokenizer_pad_token_id,
    get_tokenizer_special_token_id,
)
from mopforge.generation.kv_cache import (
    kv_cache_decode_token,
    kv_cache_prefill,
    supports_kv_cache,
)


def generate_greedy(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    device: str | None = None,
    active_modules: list[str] | None = None,
    active_adapters: list[str] | None = None,
    active_conditions: list[str] | None = None,
    use_kv_cache: bool = True,
) -> str:
    """Generate a short greedy continuation from a tiny causal-LM model.

    The function is CPU-safe by default, uses ``torch.no_grad()``, stops at EOS,
    and feeds at most ``model.max_seq_len`` tokens to the model at each step.
    """

    try:
        import torch
    except Exception as exc:
        raise ImportError("PyTorch is required for generate_greedy.") from exc

    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer.")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    target_device = torch.device(device or "cpu")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = model.to(target_device)
    model.eval()

    token_ids: list[int] = []
    bos_token_id = get_tokenizer_special_token_id(tokenizer, "bos_token_id")
    if bos_token_id is not None:
        token_ids.append(bos_token_id)
    token_ids.extend(tokenizer.encode(prompt, add_special_tokens=False))
    if not token_ids:
        token_ids.append(get_tokenizer_pad_token_id(tokenizer))
    max_seq_len = int(getattr(model, "max_seq_len", len(token_ids) + max_new_tokens))
    max_seq_len = max(1, max_seq_len)
    generated_ids: list[int] = []

    native_cache_supported = model.__class__.__name__ == "ProductionCausalLM"
    cache_supported, _ = supports_kv_cache(model)
    cache_enabled = bool(
        use_kv_cache
        and (cache_supported or native_cache_supported)
        and len(token_ids) + max_new_tokens <= max_seq_len
    )

    with torch.no_grad():
        cached_logits = None
        kv_cache = None
        if cache_enabled:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=target_device)
            if native_cache_supported:
                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "use_cache": True,
                }
                if active_modules is not None:
                    kwargs["active_modules"] = active_modules
                if active_adapters is not None:
                    kwargs["active_adapters"] = active_adapters
                if active_conditions is not None:
                    kwargs["active_conditions"] = active_conditions
                outputs = model(**kwargs)
                cached_logits = outputs["logits"]
                kv_cache = outputs["past_key_values"]
            else:
                cached_logits, kv_cache, _ = kv_cache_prefill(
                    model,
                    input_ids,
                    active_modules=active_modules,
                    active_adapters=active_adapters,
                    active_conditions=active_conditions,
                )
        for _ in range(max_new_tokens):
            if cache_enabled:
                logits = cached_logits
            else:
                window = token_ids[-max_seq_len:]
                input_ids = torch.tensor([window], dtype=torch.long, device=target_device)
                attention_mask = torch.ones_like(input_ids)
                kwargs = {}
                if active_modules is not None:
                    kwargs["active_modules"] = active_modules
                if active_adapters is not None:
                    kwargs["active_adapters"] = active_adapters
                if active_conditions is not None:
                    kwargs["active_conditions"] = active_conditions
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
                logits = outputs["logits"]
            next_id = int(logits[0, -1].argmax().item())
            if next_id == tokenizer.eos_token_id:
                break
            generated_ids.append(next_id)
            token_ids.append(next_id)
            if cache_enabled and len(generated_ids) < max_new_tokens:
                decode_ids = torch.tensor([[next_id]], dtype=torch.long, device=target_device)
                if native_cache_supported:
                    kwargs = {
                        "input_ids": decode_ids,
                        "attention_mask": torch.ones_like(decode_ids),
                        "past_key_values": kv_cache,
                        "use_cache": True,
                    }
                    if active_modules is not None:
                        kwargs["active_modules"] = active_modules
                    if active_adapters is not None:
                        kwargs["active_adapters"] = active_adapters
                    if active_conditions is not None:
                        kwargs["active_conditions"] = active_conditions
                    outputs = model(**kwargs)
                    cached_logits = outputs["logits"]
                    kv_cache = outputs["past_key_values"]
                else:
                    cached_logits, kv_cache, _ = kv_cache_decode_token(
                        model,
                        decode_ids,
                        kv_cache,
                        position=len(token_ids) - 1,
                        active_modules=active_modules,
                        active_adapters=active_adapters,
                        active_conditions=active_conditions,
                    )

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
