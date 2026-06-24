"""Production-oriented causal decoder with dense and routed-MoP MLPs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mopforge.models.fast_adapters import (
    FastAdapterBank,
    FastAdapterConfig,
    normalize_adapter_names,
)
from mopforge.models.generated_params import GeneratedAdapter, GeneratedParameterConfig


@dataclass(slots=True)
class ProductionDecoderConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_key_value_heads: int
    intermediate_size: int
    max_seq_len: int
    model_type: str = "dense"
    module_names: tuple[str, ...] = ("coding", "debugging", "tests", "repair")
    active_experts: int = 1
    routing_granularity: str = "token"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_key_value_heads",
            "intermediate_size",
            "max_seq_len",
            "active_experts",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_heads % self.n_key_value_heads:
            raise ValueError("n_heads must be divisible by n_key_value_heads.")
        if self.model_type not in {"dense", "mop_oracle", "mop_learned_router", "baseline_moe"}:
            raise ValueError("model_type is not supported by the production decoder.")
        if not self.module_names or len(set(self.module_names)) != len(self.module_names):
            raise ValueError("module_names must be non-empty and unique.")
        if self.active_experts > len(self.module_names) and self.model_type != "dense":
            raise ValueError("active_experts cannot exceed module_names.")
        if self.routing_granularity not in {"example", "token"}:
            raise ValueError("routing_granularity must be example or token.")


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except Exception:
    torch = None
    nn = None
    F = None
    ProductionCausalLM = None
    RMSNorm = None
else:
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = float(eps)

        def forward(self, hidden_states):
            dtype = hidden_states.dtype
            values = hidden_states.float()
            variance = values.pow(2).mean(dim=-1, keepdim=True)
            values = values * torch.rsqrt(variance + self.eps)
            return self.weight * values.to(dtype=dtype)


    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
            super().__init__()
            if head_dim % 2:
                raise ValueError("RoPE head_dim must be even.")
            inverse_frequency = 1.0 / (
                float(theta)
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            positions = torch.arange(max_seq_len, dtype=torch.float32)
            frequencies = torch.outer(positions, inverse_frequency)
            self.register_buffer("cos_cache", frequencies.cos(), persistent=False)
            self.register_buffer("sin_cache", frequencies.sin(), persistent=False)

        def forward(self, query, key, positions):
            cos = self.cos_cache.index_select(0, positions).to(dtype=query.dtype)
            sin = self.sin_cache.index_select(0, positions).to(dtype=query.dtype)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


    class GroupedQueryAttention(nn.Module):
        def __init__(self, config: ProductionDecoderConfig) -> None:
            super().__init__()
            self.n_heads = config.n_heads
            self.n_key_value_heads = config.n_key_value_heads
            self.head_dim = config.d_model // config.n_heads
            self.dropout = float(config.attention_dropout)
            kv_size = self.n_key_value_heads * self.head_dim
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, kv_size, bias=False)
            self.v_proj = nn.Linear(config.d_model, kv_size, bias=False)
            self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.rope = RotaryEmbedding(
                self.head_dim,
                config.max_seq_len,
                config.rope_theta,
            )

        def forward(
            self,
            hidden_states,
            *,
            attention_mask=None,
            past_key_value=None,
            use_cache: bool = False,
        ):
            batch_size, query_len, _ = hidden_states.shape
            past_len = 0 if past_key_value is None else int(past_key_value[0].shape[2])
            positions = torch.arange(
                past_len,
                past_len + query_len,
                device=hidden_states.device,
            )
            query = self._split(self.q_proj(hidden_states), self.n_heads)
            key = self._split(self.k_proj(hidden_states), self.n_key_value_heads)
            value = self._split(self.v_proj(hidden_states), self.n_key_value_heads)
            query, key = self.rope(query, key, positions)
            if past_key_value is not None:
                key = torch.cat((past_key_value[0], key), dim=2)
                value = torch.cat((past_key_value[1], value), dim=2)
            present = (key, value) if use_cache else None
            key_for_attention = _repeat_key_value(key, self.n_heads // self.n_key_value_heads)
            value_for_attention = _repeat_key_value(value, self.n_heads // self.n_key_value_heads)
            mask = _attention_mask(
                attention_mask,
                batch_size=batch_size,
                query_len=query_len,
                key_len=key_for_attention.shape[2],
                past_len=past_len,
                dtype=query.dtype,
                device=query.device,
            )
            is_causal = mask is None and past_len == 0 and query_len > 1
            attended = F.scaled_dot_product_attention(
                query,
                key_for_attention,
                value_for_attention,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attended = attended.transpose(1, 2).contiguous().view(
                batch_size,
                query_len,
                self.n_heads * self.head_dim,
            )
            return self.o_proj(attended), present

        def _split(self, values, heads: int):
            batch_size, seq_len, _ = values.shape
            return values.view(batch_size, seq_len, heads, self.head_dim).transpose(1, 2)


    class SwiGLU(nn.Module):
        def __init__(self, d_model: int, intermediate_size: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

        def forward(self, hidden_states):
            return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


    class RoutedSwiGLU(nn.Module):
        def __init__(self, config: ProductionDecoderConfig) -> None:
            super().__init__()
            self.module_names = list(config.module_names)
            self.active_experts = int(config.active_experts)
            self.routing_granularity = config.routing_granularity
            self.experts = nn.ModuleDict(
                {
                    name: SwiGLU(config.d_model, config.intermediate_size)
                    for name in self.module_names
                }
            )
            self.router = nn.Linear(config.d_model, len(self.module_names), bias=False)
            self.last_routing_metadata: dict[str, Any] = {}

        def forward(self, hidden_states, *, active_modules=None, learned_router=False):
            if learned_router:
                output, counts = self._learned_route(hidden_states)
                mode = "learned_token"
            else:
                output, counts = self._oracle_route(hidden_states, active_modules)
                mode = "oracle_example"
            total = max(1, hidden_states.shape[0] * hidden_states.shape[1])
            self.last_routing_metadata = {
                "routing_mode": mode,
                "active_expert_count": len([count for count in counts.values() if count]),
                "expert_selection_counts": counts,
                "routed_token_assignments": sum(counts.values()),
                "valid_token_count": total,
                "routing_density": sum(counts.values())
                / max(1, total * len(self.module_names)),
            }
            return output

        def _oracle_route(self, hidden_states, active_modules):
            chunks = []
            counts = {name: 0 for name in self.module_names}
            for index in range(hidden_states.shape[0]):
                requested = list(active_modules[index] if active_modules else [])
                names = [name for name in requested if name in self.experts]
                names = (names or self.module_names)[: self.active_experts]
                outputs = [self.experts[name](hidden_states[index : index + 1]) for name in names]
                chunks.append(torch.stack(outputs).mean(dim=0))
                for name in names:
                    counts[name] += int(hidden_states.shape[1])
            return torch.cat(chunks, dim=0), counts

        def _learned_route(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = self.router(flat)
            top_values, top_indices = torch.topk(
                logits,
                k=min(self.active_experts, len(self.module_names)),
                dim=-1,
            )
            weights = torch.softmax(top_values.float(), dim=-1).to(dtype=flat.dtype)
            output = torch.zeros_like(flat)
            counts = {name: 0 for name in self.module_names}
            for rank in range(top_indices.shape[1]):
                selected = top_indices[:, rank]
                for expert_index, name in enumerate(self.module_names):
                    token_indices = torch.nonzero(selected == expert_index, as_tuple=False).flatten()
                    if token_indices.numel() == 0:
                        continue
                    expert_input = flat.index_select(0, token_indices)
                    expert_output = self.experts[name](expert_input)
                    token_weights = weights[:, rank].index_select(0, token_indices).unsqueeze(-1)
                    output.index_add_(0, token_indices, expert_output * token_weights)
                    counts[name] += int(token_indices.numel())
            return output.view_as(hidden_states), counts


    class ProductionDecoderBlock(nn.Module):
        def __init__(self, config: ProductionDecoderConfig, *, routed: bool) -> None:
            super().__init__()
            self.input_norm = RMSNorm(config.d_model, config.rms_norm_eps)
            self.self_attn = GroupedQueryAttention(config)
            self.post_attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
            self.mlp = RoutedSwiGLU(config) if routed else SwiGLU(
                config.d_model,
                config.intermediate_size,
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            hidden_states,
            *,
            attention_mask=None,
            past_key_value=None,
            use_cache=False,
            active_modules=None,
            learned_router=False,
        ):
            attention, present = self.self_attn(
                self.input_norm(hidden_states),
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = hidden_states + self.dropout(attention)
            normalized = self.post_attention_norm(hidden_states)
            if isinstance(self.mlp, RoutedSwiGLU):
                feed_forward = self.mlp(
                    normalized,
                    active_modules=active_modules,
                    learned_router=learned_router,
                )
            else:
                feed_forward = self.mlp(normalized)
            return hidden_states + self.dropout(feed_forward), present


    class ProductionCausalLM(nn.Module):
        """Modern decoder-only LM compatible with MoP-Forge sparse-tail APIs."""

        def __init__(
            self,
            config: ProductionDecoderConfig,
            *,
            use_fast_adapters: bool = False,
            fast_adapter_names: Sequence[str] | None = None,
            fast_adapter_bottleneck_dim: int = 128,
            use_generated_params: bool = False,
            generated_condition_names: Sequence[str] | None = None,
            generated_condition_dim: int = 32,
            generated_rank: int = 4,
            generated_type: str = "low_rank_adapter",
        ) -> None:
            super().__init__()
            self.config = config
            self.max_seq_len = config.max_seq_len
            self.model_type = config.model_type
            self.module_names = list(config.module_names)
            self.active_experts = config.active_experts
            self.routing_granularity = config.routing_granularity
            self.activation_checkpointing_enabled = False
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            routed = config.model_type != "dense"
            layers = nn.ModuleList(
                [ProductionDecoderBlock(config, routed=routed) for _ in range(config.n_layers)]
            )
            if routed:
                self.routed_blocks = layers
                self.blocks = None
            else:
                self.blocks = layers
                self.routed_blocks = None
            self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_word_embeddings:
                self.lm_head.weight = self.token_embedding.weight
            self.use_fast_adapters = bool(use_fast_adapters)
            self.fast_adapter_names = list(fast_adapter_names or ["default"])
            self.fast_adapter_bank = None
            if self.use_fast_adapters:
                self.fast_adapter_bank = FastAdapterBank(
                    FastAdapterConfig(
                        d_model=config.d_model,
                        bottleneck_dim=fast_adapter_bottleneck_dim,
                        adapter_names=self.fast_adapter_names,
                        dropout=config.dropout,
                    )
                )
            self.generated_adapter = None
            self.generated_condition_names = list(generated_condition_names or ["default"])
            if use_generated_params:
                self.generated_adapter = GeneratedAdapter(
                    GeneratedParameterConfig(
                        d_model=config.d_model,
                        condition_dim=generated_condition_dim,
                        rank=generated_rank,
                        generator_type=generated_type,
                        condition_names=self.generated_condition_names,
                    )
                )
            self.last_forward_metadata: dict[str, Any] = {}
            self.apply(self._init_weights)

        @property
        def layers(self):
            return self.routed_blocks if self.routed_blocks is not None else self.blocks

        def forward(
            self,
            input_ids=None,
            hidden_states=None,
            attention_mask=None,
            labels=None,
            active_modules=None,
            active_adapters=None,
            active_conditions=None,
            past_key_values=None,
            use_cache: bool = False,
        ):
            if hidden_states is not None:
                if input_ids is not None:
                    raise ValueError("Pass input_ids or hidden_states, not both.")
                output = self.forward_from_hidden(
                    hidden_states,
                    attention_mask=attention_mask,
                    labels=labels,
                    active_modules=active_modules,
                    active_adapters=active_adapters,
                    active_conditions=active_conditions,
                )
                output["past_key_values"] = None
                return output
            if input_ids is None:
                raise ValueError("input_ids or hidden_states is required.")
            encoded = self.encode_for_sparse_tail(
                input_ids,
                attention_mask=attention_mask,
                active_modules=active_modules,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            output = self.forward_from_hidden(
                encoded["hidden_states"],
                attention_mask=attention_mask,
                labels=labels,
                active_modules=encoded["active_modules"],
                active_adapters=active_adapters,
                active_conditions=active_conditions,
            )
            output["past_key_values"] = encoded["past_key_values"]
            return output

        def encode_for_sparse_tail(
            self,
            input_ids,
            *,
            attention_mask=None,
            active_modules=None,
            past_key_values=None,
            use_cache: bool = False,
        ):
            batch_size, seq_len = input_ids.shape
            past_len = 0 if not past_key_values else int(past_key_values[0][0].shape[2])
            if past_len + seq_len > self.max_seq_len:
                raise ValueError("Sequence plus K/V cache exceeds max_seq_len.")
            modules = self._expand_active_modules(active_modules, batch_size)
            hidden_states = self.token_embedding(input_ids)
            next_cache = []
            learned_router = self.model_type == "mop_learned_router"
            for index, block in enumerate(self.layers):
                past = None if past_key_values is None else past_key_values[index]
                if self.activation_checkpointing_enabled and self.training and not use_cache:
                    hidden_states = activation_checkpoint(
                        lambda hidden, current=block: current(
                            hidden,
                            attention_mask=attention_mask,
                            active_modules=modules,
                            learned_router=learned_router,
                        )[0],
                        hidden_states,
                        use_reentrant=False,
                    )
                    present = None
                else:
                    hidden_states, present = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_value=past,
                        use_cache=use_cache,
                        active_modules=modules,
                        learned_router=learned_router,
                    )
                if use_cache:
                    next_cache.append(present)
            routed_metadata = []
            if self.routed_blocks is not None:
                routed_metadata = [
                    dict(block.mlp.last_routing_metadata)
                    for block in self.routed_blocks
                ]
            self.last_forward_metadata = {
                "architecture_family": "production_decoder_v2",
                "native_kv_cache": bool(use_cache),
                "past_tokens": past_len,
                "routed_block_metadata": routed_metadata,
                "tail_from_hidden": False,
            }
            return {
                "hidden_states": hidden_states,
                "active_modules": modules,
                "past_key_values": tuple(next_cache) if use_cache else None,
                "metadata": dict(self.last_forward_metadata),
            }

        def forward_from_hidden(
            self,
            hidden_states,
            *,
            attention_mask=None,
            labels=None,
            active_modules=None,
            active_adapters=None,
            active_conditions=None,
        ):
            batch_size = hidden_states.shape[0]
            adapters = self._expand_active_adapters(active_adapters, batch_size)
            if self.fast_adapter_bank is not None:
                hidden_states = self.fast_adapter_bank(
                    hidden_states,
                    active_adapters=adapters,
                )
            conditions = self._expand_active_conditions(active_conditions, batch_size)
            if self.generated_adapter is not None:
                hidden_states = self.generated_adapter(
                    hidden_states,
                    active_conditions=conditions,
                )
            logits = self.lm_head(self.norm(hidden_states))
            loss = _causal_loss(logits, labels)
            self.last_forward_metadata["tail_from_hidden"] = True
            return {
                "logits": logits,
                "loss": loss,
                "active_modules": self._expand_active_modules(active_modules, batch_size),
                "active_adapters": adapters,
                "active_conditions": conditions,
            }

        def _expand_active_modules(self, active_modules, batch_size: int):
            if active_modules is None:
                return [[] for _ in range(batch_size)]
            values = list(active_modules) if not isinstance(active_modules, str) else [active_modules]
            if not values:
                return [[] for _ in range(batch_size)]
            if all(isinstance(value, str) for value in values):
                names = [value for value in values if value in self.module_names]
                return [names for _ in range(batch_size)]
            if len(values) != batch_size:
                raise ValueError("Per-example active_modules length must match batch size.")
            return [
                [name for name in names if name in self.module_names]
                for names in values
            ]

        def _expand_active_adapters(self, active_adapters, batch_size: int):
            if self.fast_adapter_bank is None or active_adapters is None:
                return [[] for _ in range(batch_size)]
            values = list(active_adapters) if not isinstance(active_adapters, str) else [active_adapters]
            if all(isinstance(value, str) for value in values):
                names = normalize_adapter_names(values, self.fast_adapter_names)
                return [names for _ in range(batch_size)]
            if len(values) != batch_size:
                raise ValueError("Per-example active_adapters length must match batch size.")
            return [normalize_adapter_names(names, self.fast_adapter_names) for names in values]

        def _expand_active_conditions(self, active_conditions, batch_size: int):
            if self.generated_adapter is None:
                return [[] for _ in range(batch_size)]
            return self.generated_adapter.expand_active_conditions(active_conditions, batch_size)

        @staticmethod
        def _init_weights(module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _apply_rope(values, cos, sin):
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    return torch.cat(
        (first * cos - second * sin, first * sin + second * cos),
        dim=-1,
    )


def _repeat_key_value(values, repeats: int):
    if repeats == 1:
        return values
    return values.repeat_interleave(repeats, dim=1)


def _attention_mask(
    attention_mask,
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    past_len: int,
    dtype,
    device,
):
    if attention_mask is None and (past_len == 0 or query_len == 1):
        return None
    minimum = torch.finfo(dtype).min
    allowed = torch.ones((batch_size, 1, query_len, key_len), dtype=torch.bool, device=device)
    query_positions = torch.arange(past_len, past_len + query_len, device=device).view(1, 1, -1, 1)
    key_positions = torch.arange(key_len, device=device).view(1, 1, 1, -1)
    allowed &= key_positions <= query_positions
    if attention_mask is not None:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.shape[1] != key_len:
            if mask.shape[1] == query_len and past_len:
                prefix = torch.ones((batch_size, past_len), dtype=torch.bool, device=device)
                mask = torch.cat((prefix, mask), dim=1)
            else:
                raise ValueError("attention_mask length does not match current plus cached tokens.")
        allowed &= mask[:, None, None, :]
    additive = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return additive.masked_fill(~allowed, minimum)


def _causal_loss(logits, labels):
    if labels is None:
        return None
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    if not (shifted_labels != -100).any():
        return logits.sum() * 0.0
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
    )


def production_parameter_count(config: ProductionDecoderConfig) -> int:
    """Return an exact analytic parameter count without allocating the model."""

    d_model = config.d_model
    head_dim = d_model // config.n_heads
    kv_size = config.n_key_value_heads * head_dim
    attention = (2 * d_model * d_model) + (2 * d_model * kv_size)
    norms = 2 * d_model
    expert_count = len(config.module_names) if config.model_type != "dense" else 1
    mlp = expert_count * 3 * d_model * config.intermediate_size
    router = d_model * expert_count if config.model_type != "dense" else 0
    blocks = config.n_layers * (attention + norms + mlp + router)
    embeddings = config.vocab_size * d_model
    head = 0 if config.tie_word_embeddings else config.vocab_size * d_model
    return int(embeddings + blocks + d_model + head)
