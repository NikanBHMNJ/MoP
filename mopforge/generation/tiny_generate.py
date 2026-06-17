"""Greedy generation for tiny MoP-Forge causal-LM smoke models."""

from __future__ import annotations

from typing import Any


def generate_greedy(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    device: str | None = None,
    active_modules: list[str] | None = None,
    active_adapters: list[str] | None = None,
    active_conditions: list[str] | None = None,
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

    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    max_seq_len = int(getattr(model, "max_seq_len", len(token_ids) + max_new_tokens))
    max_seq_len = max(1, max_seq_len)
    generated_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
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
            next_id = int(outputs["logits"][0, -1].argmax().item())
            if next_id == tokenizer.eos_token_id:
                break
            generated_ids.append(next_id)
            token_ids.append(next_id)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
