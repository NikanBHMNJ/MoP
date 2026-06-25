"""Optional tiny oracle-routed Mixture-of-Parameters causal LM."""

from __future__ import annotations

from collections.abc import Sequence

from mopforge.models.fast_adapters import (
    FastAdapterBank,
    FastAdapterConfig,
    normalize_adapter_names,
)
from mopforge.models.generated_params import GeneratedAdapter, GeneratedParameterConfig
from mopforge.training.routing import normalize_target_modules


try:
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except Exception:
    torch = None
    nn = None
    TinyMoPCausalTransformer = None
else:

    class ModuleMLPBlock(nn.Module):
        """A tiny module-specific residual MLP block."""

        def __init__(self, d_model: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, hidden_states):
            """Apply a residual module-specific transformation."""

            return self.net(hidden_states)


    class ModuleBank(nn.Module):
        """Collection of named module-specific parameter blocks."""

        def __init__(self, module_names: Sequence[str], d_model: int, dropout: float) -> None:
            super().__init__()
            self.module_names = list(module_names)
            self.blocks = nn.ModuleDict(
                {
                    module_name: ModuleMLPBlock(d_model, dropout=dropout)
                    for module_name in self.module_names
                }
            )

        def forward_one(self, hidden_states, module_names: Sequence[str]):
            """Average outputs from the selected module blocks."""

            outputs = [self.blocks[module_name](hidden_states) for module_name in module_names]
            if not outputs:
                return torch.zeros_like(hidden_states)
            return torch.stack(outputs, dim=0).mean(dim=0)


    class RoutedFFNBlock(nn.Module):
        """Transformer block with shared attention and routed FFN experts."""

        def __init__(
            self,
            module_names: Sequence[str],
            d_model: int,
            n_heads: int,
            dropout: float,
            routing_granularity: str = "example",
        ) -> None:
            super().__init__()
            self.module_names = list(module_names)
            self.routing_granularity = routing_granularity
            self.attn_norm = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_dropout = nn.Dropout(dropout)
            self.ffn_norm = nn.LayerNorm(d_model)
            self.router = nn.Linear(d_model, len(self.module_names), bias=False)
            self.experts = nn.ModuleDict(
                {
                    module_name: nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model * 4, d_model),
                        nn.Dropout(dropout),
                    )
                    for module_name in self.module_names
                }
            )
            self.last_routing_metadata: dict[str, object] = {}

        def forward(
            self,
            hidden_states,
            *,
            attention_mask=None,
            active_modules: Sequence[Sequence[str]] | None = None,
            active_experts: int = 1,
            routing_granularity: str | None = None,
        ):
            granularity = routing_granularity or self.routing_granularity
            if granularity not in {"example", "token"}:
                raise ValueError("routing_granularity must be example or token.")
            seq_len = hidden_states.shape[1]
            device = hidden_states.device
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            padding_mask = attention_mask == 0 if attention_mask is not None else None
            attn_output, _ = self.self_attn(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            hidden_states = self.attn_norm(
                hidden_states + self.attn_dropout(attn_output)
            )
            if granularity == "token":
                routed_output, metadata = self._route_tokens(
                    hidden_states,
                    attention_mask=attention_mask,
                    active_modules=active_modules,
                    active_experts=active_experts,
                )
            else:
                routed_output, metadata = self._route_examples(
                    hidden_states,
                    attention_mask=attention_mask,
                    active_modules=active_modules,
                    active_experts=active_experts,
                )
            self.last_routing_metadata = metadata
            return self.ffn_norm(hidden_states + routed_output)

        def _route_examples(
            self,
            hidden_states,
            *,
            attention_mask,
            active_modules,
            active_experts: int,
        ):
            routed_chunks = []
            selection_counts = {name: 0 for name in self.module_names}
            routed_assignments = 0
            for index in range(hidden_states.shape[0]):
                names = self._candidate_names(active_modules, index)
                names = names[: min(len(names), max(1, int(active_experts)))]
                outputs = [
                    self.experts[name](hidden_states[index : index + 1])
                    for name in names
                ]
                routed_chunks.append(torch.stack(outputs, dim=0).mean(dim=0))
                valid_tokens = self._valid_token_count(
                    attention_mask,
                    index,
                    hidden_states.shape[1],
                )
                routed_assignments += valid_tokens * len(names)
                for name in names:
                    selection_counts[name] += valid_tokens
            routed = torch.cat(routed_chunks, dim=0)
            return routed, self._routing_metadata(
                "example",
                selection_counts,
                routed_assignments,
                attention_mask,
                hidden_states,
            )

        def _route_tokens(
            self,
            hidden_states,
            *,
            attention_mask,
            active_modules,
            active_experts: int,
        ):
            routed = torch.zeros_like(hidden_states)
            selection_counts = {name: 0 for name in self.module_names}
            routed_assignments = 0
            for batch_index in range(hidden_states.shape[0]):
                names = self._candidate_names(active_modules, batch_index)
                expert_indices = torch.tensor(
                    [self.module_names.index(name) for name in names],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                logits = self.router(hidden_states[batch_index]).index_select(
                    -1,
                    expert_indices,
                )
                probabilities = torch.softmax(logits, dim=-1)
                top_k = min(len(names), max(1, int(active_experts)))
                top_probabilities, top_positions = torch.topk(
                    probabilities,
                    k=top_k,
                    dim=-1,
                )
                normalized = top_probabilities / top_probabilities.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-9)
                # Preserve normalized top-k outputs while retaining a useful
                # main-loss gradient for a top-1 learned router.
                routing_weights = (
                    normalized
                    + top_probabilities
                    - top_probabilities.detach()
                )
                valid = self._valid_token_mask(
                    attention_mask,
                    batch_index,
                    hidden_states.shape[1],
                    hidden_states.device,
                )
                for rank in range(top_k):
                    selected_positions = top_positions[:, rank]
                    for candidate_index, name in enumerate(names):
                        selected = valid & (selected_positions == candidate_index)
                        token_indices = torch.nonzero(selected, as_tuple=False).flatten()
                        if token_indices.numel() == 0:
                            continue
                        expert_input = hidden_states[batch_index].index_select(
                            0,
                            token_indices,
                        )
                        expert_output = self.experts[name](expert_input)
                        weights = routing_weights[:, rank].index_select(
                            0,
                            token_indices,
                        ).unsqueeze(-1)
                        routed[batch_index].index_add_(
                            0,
                            token_indices,
                            expert_output * weights,
                        )
                        selected_count = int(token_indices.numel())
                        selection_counts[name] += selected_count
                        routed_assignments += selected_count
            return routed, self._routing_metadata(
                "token",
                selection_counts,
                routed_assignments,
                attention_mask,
                hidden_states,
            )

        def _candidate_names(self, active_modules, index: int) -> list[str]:
            names = list(active_modules[index] if active_modules else [])
            names = [name for name in names if name in self.experts]
            return names or list(self.module_names)

        def _valid_token_mask(self, attention_mask, index, seq_len, device):
            if attention_mask is None:
                return torch.ones(seq_len, dtype=torch.bool, device=device)
            return attention_mask[index].to(device=device, dtype=torch.bool)

        def _valid_token_count(self, attention_mask, index: int, seq_len: int) -> int:
            if attention_mask is None:
                return int(seq_len)
            return int(attention_mask[index].sum().detach().cpu().item())

        def _routing_metadata(
            self,
            granularity,
            selection_counts,
            routed_assignments,
            attention_mask,
            hidden_states,
        ) -> dict[str, object]:
            if attention_mask is None:
                valid_tokens = int(hidden_states.shape[0] * hidden_states.shape[1])
            else:
                valid_tokens = int(attention_mask.sum().detach().cpu().item())
            active_names = [
                name for name, count in selection_counts.items() if count > 0
            ]
            denominator = max(1, valid_tokens * len(self.module_names))
            return {
                "routing_granularity": granularity,
                "active_expert_names": active_names,
                "active_expert_count": len(active_names),
                "routed_token_assignments": int(routed_assignments),
                "valid_token_count": valid_tokens,
                "routing_density": float(routed_assignments) / float(denominator),
                "expert_selection_counts": {
                    name: int(count)
                    for name, count in selection_counts.items()
                    if count > 0
                },
            }


    class LoRADelta(nn.Module):
        """Small low-rank residual delta used as a LoRA-style quality valve."""

        def __init__(self, d_model: int, rank: int, dropout: float) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.down = nn.Linear(d_model, rank, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.up = nn.Linear(rank, d_model, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, hidden_states):
            return self.up(self.dropout(self.down(self.norm(hidden_states))))


    class LoRADeltaBank(nn.Module):
        """Named bank of low-rank residual deltas routed by module name."""

        def __init__(
            self,
            module_names: Sequence[str],
            d_model: int,
            rank: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.module_names = list(module_names)
            self.deltas = nn.ModuleDict(
                {
                    module_name: LoRADelta(d_model, rank=rank, dropout=dropout)
                    for module_name in self.module_names
                }
            )

        def forward(self, hidden_states, active_modules=None):
            if active_modules is None:
                return hidden_states
            if isinstance(active_modules, str):
                active_modules = [[active_modules] for _ in range(hidden_states.shape[0])]
            elif active_modules and all(isinstance(item, str) for item in list(active_modules)):
                active_modules = [list(active_modules) for _ in range(hidden_states.shape[0])]
            chunks = []
            for index, names in enumerate(active_modules or []):
                normalized = [name for name in names if name in self.deltas]
                if not normalized:
                    chunks.append(hidden_states[index : index + 1])
                    continue
                deltas = [
                    self.deltas[name](hidden_states[index : index + 1])
                    for name in normalized
                ]
                chunks.append(
                    hidden_states[index : index + 1]
                    + torch.stack(deltas, dim=0).mean(dim=0)
                )
            if not chunks:
                return hidden_states
            return torch.cat(chunks, dim=0)


    class LowRankProjection(nn.Module):
        """Zero-initialized low-rank additive projection."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.down = nn.Linear(in_features, rank, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.up = nn.Linear(rank, out_features, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, hidden_states):
            return self.up(self.dropout(self.down(hidden_states)))


    class RoutedLowRankProjectionBank(nn.Module):
        """Module-routed low-rank deltas for one transformer projection."""

        def __init__(
            self,
            module_names: Sequence[str],
            in_features: int,
            out_features: int,
            rank: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.deltas = nn.ModuleDict(
                {
                    name: LowRankProjection(
                        in_features,
                        out_features,
                        rank,
                        dropout,
                    )
                    for name in module_names
                }
            )

        def forward(self, hidden_states, active_modules=None):
            if active_modules is None:
                return torch.zeros(
                    (*hidden_states.shape[:-1], self._out_features()),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            chunks = []
            for index in range(hidden_states.shape[0]):
                names = list(active_modules[index] if active_modules else [])
                names = [name for name in names if name in self.deltas]
                if not names:
                    chunks.append(
                        torch.zeros(
                            (
                                1,
                                hidden_states.shape[1],
                                self._out_features(),
                            ),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                    )
                    continue
                outputs = [
                    self.deltas[name](hidden_states[index : index + 1])
                    for name in names
                ]
                chunks.append(torch.stack(outputs, dim=0).mean(dim=0))
            return torch.cat(chunks, dim=0)

        def _out_features(self) -> int:
            first = next(iter(self.deltas.values()))
            return int(first.up.out_features)


    class RoutedLoRATransformerEncoderLayer(nn.TransformerEncoderLayer):
        """Encoder layer with routed low-rank Q/K/V, output, and FFN deltas."""

        def __init__(
            self,
            *,
            module_names: Sequence[str],
            lora_rank: int,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
        ) -> None:
            super().__init__(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            bank_kwargs = {
                "module_names": module_names,
                "rank": lora_rank,
                "dropout": dropout,
            }
            self.q_lora_bank = RoutedLowRankProjectionBank(
                in_features=d_model,
                out_features=d_model,
                **bank_kwargs,
            )
            self.k_lora_bank = RoutedLowRankProjectionBank(
                in_features=d_model,
                out_features=d_model,
                **bank_kwargs,
            )
            self.v_lora_bank = RoutedLowRankProjectionBank(
                in_features=d_model,
                out_features=d_model,
                **bank_kwargs,
            )
            self.attn_out_lora_bank = RoutedLowRankProjectionBank(
                in_features=d_model,
                out_features=d_model,
                **bank_kwargs,
            )
            self.ffn_up_lora_bank = RoutedLowRankProjectionBank(
                in_features=d_model,
                out_features=dim_feedforward,
                **bank_kwargs,
            )
            self.ffn_down_lora_bank = RoutedLowRankProjectionBank(
                in_features=dim_feedforward,
                out_features=d_model,
                **bank_kwargs,
            )
            self._active_modules = None

        def set_active_modules(self, active_modules) -> None:
            self._active_modules = active_modules

        def forward(
            self,
            src,
            src_mask=None,
            src_key_padding_mask=None,
            is_causal: bool = False,
        ):
            hidden_states = src
            if self.norm_first:
                hidden_states = hidden_states + self._sa_block(
                    self.norm1(hidden_states),
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                )
                hidden_states = hidden_states + self._ff_block(
                    self.norm2(hidden_states)
                )
                return hidden_states
            hidden_states = self.norm1(
                hidden_states
                + self._sa_block(
                    hidden_states,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            return self.norm2(hidden_states + self._ff_block(hidden_states))

        def _sa_block(
            self,
            hidden_states,
            attention_mask,
            key_padding_mask,
            is_causal: bool = False,
        ):
            query = hidden_states + self.q_lora_bank(
                hidden_states,
                self._active_modules,
            )
            key = hidden_states + self.k_lora_bank(
                hidden_states,
                self._active_modules,
            )
            value = hidden_states + self.v_lora_bank(
                hidden_states,
                self._active_modules,
            )
            output = self.self_attn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            output = output + self.attn_out_lora_bank(
                output,
                self._active_modules,
            )
            return self.dropout1(output)

        def _ff_block(self, hidden_states):
            projected = self.linear1(hidden_states) + self.ffn_up_lora_bank(
                hidden_states,
                self._active_modules,
            )
            activated = self.dropout(self.activation(projected))
            output = self.linear2(activated) + self.ffn_down_lora_bank(
                activated,
                self._active_modules,
            )
            return self.dropout2(output)


    class TinyMoPCausalTransformer(nn.Module):
        """Tiny oracle-routed MoP causal LM for CPU smoke tests.

        Routing is oracle-supplied for this tiny model. The caller supplies active modules,
        usually from ``KnowledgeLesson.target_modules``. Backward-compatible
        configs include ``core`` automatically; newer sparse efficiency configs
        can set ``always_include_core=False`` and omit redundant core modules.
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 64,
            n_heads: int = 2,
            n_layers: int = 2,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            module_names: Sequence[str] | None = None,
            always_include_core: bool = True,
            mop_block_type: str = "post_core_mlp",
            expert_count: int | None = None,
            active_experts: int = 1,
            routing_granularity: str = "example",
            shared_depth_ratio: float = 1.0,
            use_lora_deltas: bool = False,
            lora_tail_only: bool = False,
            lora_rank: int = 0,
            lora_target_modules: Sequence[str] | None = None,
            use_fast_adapters: bool = False,
            fast_adapter_names: Sequence[str] | None = None,
            fast_adapter_bottleneck_dim: int = 16,
            use_generated_params: bool = False,
            generated_condition_names: Sequence[str] | None = None,
            generated_condition_dim: int = 32,
            generated_rank: int = 4,
            generated_type: str = "low_rank_adapter",
        ) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads.")
            if max_seq_len <= 0:
                raise ValueError("max_seq_len must be positive.")

            self.module_names = list(
                module_names or ["core", "coding", "debugging", "math", "planning"]
            )
            if not self.module_names:
                raise ValueError("module_names must contain at least one module.")
            if len(self.module_names) != len(set(self.module_names)):
                raise ValueError("module_names must not contain duplicates.")
            if mop_block_type not in {"post_core_mlp", "routed_ffn"}:
                raise ValueError("mop_block_type must be post_core_mlp or routed_ffn.")
            if active_experts <= 0:
                raise ValueError("active_experts must be positive.")
            if routing_granularity not in {"example", "token"}:
                raise ValueError("routing_granularity must be example or token.")
            if not 0.0 < float(shared_depth_ratio) <= 1.0:
                raise ValueError("shared_depth_ratio must be in (0.0, 1.0].")
            if use_lora_deltas and lora_rank <= 0:
                raise ValueError("lora_rank must be positive when use_lora_deltas is true.")
            if lora_tail_only and not use_lora_deltas:
                raise ValueError("lora_tail_only requires use_lora_deltas=true.")

            self.max_seq_len = max_seq_len
            self.always_include_core = bool(always_include_core)
            self.mop_block_type = mop_block_type
            self.expert_count = int(expert_count or len(self.module_names))
            self.active_experts = int(active_experts)
            self.routing_granularity = routing_granularity
            self.shared_depth_ratio = float(shared_depth_ratio)
            self.use_fast_adapters = bool(use_fast_adapters)
            self.fast_adapter_names = list(fast_adapter_names or ["default"])
            self.use_generated_params = bool(use_generated_params)
            self.generated_condition_names = list(generated_condition_names or ["default"])
            self.use_lora_deltas = bool(use_lora_deltas)
            self.lora_tail_only = bool(lora_tail_only)
            self.lora_target_modules = list(lora_target_modules or self.module_names)
            self.last_forward_metadata: dict[str, object] = {}
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation_checkpointing_enabled = False
            self.routed_expert_names = _expert_names(self.module_names, self.expert_count)
            self.routed_blocks = nn.ModuleList()
            self.shared_layer_count = n_layers
            self.routed_layer_count = 0
            self.requested_shared_depth_ratio = float(shared_depth_ratio)

            def new_encoder_layer():
                if self.use_lora_deltas and not self.lora_tail_only:
                    return RoutedLoRATransformerEncoderLayer(
                        module_names=self.lora_target_modules,
                        lora_rank=lora_rank,
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                    )
                return nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )

            if self.mop_block_type == "routed_ffn":
                self.shared_layer_count = min(
                    n_layers - 1,
                    max(0, round(n_layers * self.requested_shared_depth_ratio)),
                )
                self.routed_layer_count = n_layers - self.shared_layer_count
                self.shared_depth_ratio = (
                    float(self.shared_layer_count) / float(n_layers)
                )
                if self.shared_layer_count:
                    encoder_layer = new_encoder_layer()
                    self.shared_blocks = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=self.shared_layer_count,
                    )
                else:
                    self.shared_blocks = None
                self.routed_blocks = nn.ModuleList(
                    [
                        RoutedFFNBlock(
                            self.routed_expert_names,
                            d_model=d_model,
                            n_heads=n_heads,
                            dropout=dropout,
                            routing_granularity=routing_granularity,
                        )
                        for _ in range(self.routed_layer_count)
                    ]
                )
                self.module_bank = None
            else:
                encoder_layer = new_encoder_layer()
                self.shared_blocks = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers
                )
                self.module_bank = ModuleBank(self.module_names, d_model, dropout)
            self.last_warm_start_metadata: dict[str, object] = {}
            self.internal_lora_enabled = bool(
                self.use_lora_deltas
                and not self.lora_tail_only
                and self.shared_blocks is not None
            )
            self.lora_delta_bank = None
            if self.use_lora_deltas and not self.internal_lora_enabled:
                self.lora_delta_bank = LoRADeltaBank(
                    self.lora_target_modules,
                    d_model=d_model,
                    rank=lora_rank,
                    dropout=dropout,
                )
            self.fast_adapter_bank = None
            if self.use_fast_adapters:
                if FastAdapterBank is None:
                    raise RuntimeError("PyTorch is required for FastAdapterBank.")
                self.fast_adapter_bank = FastAdapterBank(
                    FastAdapterConfig(
                        d_model=d_model,
                        bottleneck_dim=fast_adapter_bottleneck_dim,
                        adapter_names=list(self.fast_adapter_names),
                        dropout=dropout,
                    )
                )
            self.generated_adapter = None
            if self.use_generated_params:
                if GeneratedAdapter is None:
                    raise RuntimeError("PyTorch is required for GeneratedAdapter.")
                self.generated_adapter = GeneratedAdapter(
                    GeneratedParameterConfig(
                        d_model=d_model,
                        condition_dim=generated_condition_dim,
                        rank=generated_rank,
                        generator_type=generated_type,
                        condition_names=list(self.generated_condition_names),
                    )
                )
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            active_modules: Sequence[str] | Sequence[Sequence[str]] | None = None,
            active_adapters: Sequence[str] | Sequence[Sequence[str]] | None = None,
            active_conditions: Sequence[str] | Sequence[Sequence[str]] | None = None,
        ):
            """Run a causal-LM forward pass with oracle module routing."""

            batch_size, seq_len = input_ids.shape
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_seq_len "
                    f"{self.max_seq_len}."
                )

            per_example_modules = self._expand_active_modules(
                active_modules,
                batch_size,
            )
            routed_hidden = self.encode_for_sparse_tail(
                input_ids,
                attention_mask=attention_mask,
                active_modules=per_example_modules,
            )
            return self.forward_from_hidden(
                routed_hidden["hidden_states"],
                attention_mask=attention_mask,
                labels=labels,
                active_modules=per_example_modules,
                active_adapters=active_adapters,
                active_conditions=active_conditions,
            )

        def encode_for_sparse_tail(
            self,
            input_ids,
            *,
            attention_mask=None,
            active_modules: Sequence[Sequence[str]] | None = None,
        ) -> dict[str, object]:
            """Encode token IDs up to the trainable sparse tail boundary."""

            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            per_example_modules = active_modules or self._expand_active_modules(
                None,
                batch_size,
            )
            shared_prefix_frozen = self._module_fully_frozen(
                self.token_embedding,
                self.position_embedding,
                self.shared_blocks,
            )
            shared_param_count = self._module_param_count(
                self.token_embedding,
                self.position_embedding,
                self.shared_blocks,
            )
            metadata = {
                "mop_block_type": self.mop_block_type,
                "routing_granularity": self.routing_granularity,
                "active_experts": self.active_experts,
                "shared_layer_count": self.shared_layer_count,
                "routed_layer_count": self.routed_layer_count,
                "shared_depth_ratio": self.shared_depth_ratio,
                "frozen_prefix_param_count": shared_param_count,
                "frozen_prefix_no_grad_enabled": bool(shared_prefix_frozen),
                "frozen_prefix_activation_detached": bool(shared_prefix_frozen),
            }
            if shared_prefix_frozen:
                with torch.no_grad():
                    shared_hidden = self._encode_shared_prefix(
                        input_ids,
                        attention_mask,
                        per_example_modules,
                    )
                shared_hidden = shared_hidden.detach()
            else:
                shared_hidden = self._encode_shared_prefix(
                    input_ids,
                    attention_mask,
                    per_example_modules,
                )

            routed_blocks_frozen = bool(self.routed_blocks) and self._module_fully_frozen(
                self.routed_blocks
            )
            routed_blocks_no_grad = bool(
                routed_blocks_frozen and not shared_hidden.requires_grad
            )
            metadata["frozen_routed_blocks_no_grad_enabled"] = bool(
                routed_blocks_no_grad
            )
            metadata["routed_blocks_fully_frozen"] = bool(routed_blocks_frozen)
            metadata["frozen_routed_blocks_param_count"] = self._module_param_count(
                self.routed_blocks
            )
            if self.routed_blocks:
                if routed_blocks_no_grad:
                    with torch.no_grad():
                        shared_hidden = self._apply_routed_blocks(
                            shared_hidden,
                            attention_mask,
                            per_example_modules,
                        )
                    shared_hidden = shared_hidden.detach()
                else:
                    shared_hidden = self._apply_routed_blocks(
                        shared_hidden,
                        attention_mask,
                        per_example_modules,
                    )
                metadata["routed_block_metadata"] = [
                    dict(block.last_routing_metadata)
                    for block in self.routed_blocks
                ]

            module_bank_frozen = self.module_bank is not None and self._module_fully_frozen(self.module_bank)
            module_bank_no_grad = bool(
                module_bank_frozen and not shared_hidden.requires_grad
            )
            metadata["frozen_module_bank_no_grad_enabled"] = bool(module_bank_no_grad)
            metadata["module_bank_fully_frozen"] = bool(module_bank_frozen)
            metadata["frozen_module_bank_param_count"] = (
                self._module_param_count(self.module_bank)
                if self.module_bank is not None
                else 0
            )
            if self.module_bank is not None:
                if module_bank_no_grad:
                    with torch.no_grad():
                        routed_hidden = self._apply_module_bank(
                            shared_hidden,
                            per_example_modules,
                        )
                    routed_hidden = routed_hidden.detach()
                else:
                    routed_hidden = self._apply_module_bank(
                        shared_hidden,
                        per_example_modules,
                    )
            else:
                routed_hidden = shared_hidden
            self.last_forward_metadata = metadata
            return {
                "hidden_states": routed_hidden,
                "active_modules": per_example_modules,
                "metadata": dict(metadata),
            }

        def forward_from_hidden(
            self,
            hidden_states,
            *,
            attention_mask=None,
            labels=None,
            active_modules: Sequence[str] | Sequence[Sequence[str]] | None = None,
            active_adapters: Sequence[str] | Sequence[Sequence[str]] | None = None,
            active_conditions: Sequence[str] | Sequence[Sequence[str]] | None = None,
        ):
            """Run adapters/norm/head from cached or freshly encoded hidden states."""

            batch_size = hidden_states.shape[0]
            per_example_modules = self._expand_active_modules(active_modules, batch_size)
            if hidden_states.device.type == "cpu" and hidden_states.dtype != self.norm.weight.dtype:
                hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
            if self.lora_delta_bank is not None:
                hidden_states = self.lora_delta_bank(
                    hidden_states,
                    active_modules=per_example_modules,
                )
            per_example_adapters = self._expand_active_adapters(
                active_adapters,
                batch_size,
            )
            if self.fast_adapter_bank is not None:
                hidden_states = self.fast_adapter_bank(
                    hidden_states,
                    active_adapters=per_example_adapters,
                )
            per_example_conditions = self._expand_active_conditions(
                active_conditions,
                batch_size,
            )
            if self.generated_adapter is not None:
                hidden_states = self.generated_adapter(
                    hidden_states,
                    active_conditions=per_example_conditions,
                )

            logits = self.lm_head(self.norm(hidden_states))
            loss = self._loss_from_logits(logits, labels)
            self.last_forward_metadata.update(
                {
                    "tail_from_hidden": True,
                    "lora_delta_enabled": self.use_lora_deltas,
                    "internal_lora_enabled": self.internal_lora_enabled,
                    "lora_tail_only": self.lora_tail_only,
                }
            )
            return {
                "logits": logits,
                "loss": loss,
                "active_modules": per_example_modules,
                "active_adapters": per_example_adapters,
                "active_conditions": per_example_conditions,
            }

        def _encode_shared_prefix(
            self,
            input_ids,
            attention_mask,
            per_example_modules,
        ):
            seq_len = input_ids.shape[1]
            device = input_ids.device
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
            hidden_states = self.dropout(hidden_states)
            if self.shared_blocks is None:
                return hidden_states
            for layer in self.shared_blocks.layers:
                set_active_modules = getattr(layer, "set_active_modules", None)
                if callable(set_active_modules):
                    set_active_modules(per_example_modules)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            padding_mask = attention_mask == 0 if attention_mask is not None else None
            if (
                self.activation_checkpointing_enabled
                and self.training
                and torch.is_grad_enabled()
            ):
                for layer in self.shared_blocks.layers:
                    hidden_states = activation_checkpoint(
                        lambda hidden, current_layer=layer: current_layer(
                            hidden,
                            src_mask=causal_mask,
                            src_key_padding_mask=padding_mask,
                            is_causal=True,
                        ),
                        hidden_states,
                        use_reentrant=False,
                    )
                return hidden_states
            return self.shared_blocks(
                hidden_states,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )

        def _apply_routed_blocks(
            self,
            hidden_states,
            attention_mask,
            per_example_modules,
        ):
            for block in self.routed_blocks:
                if (
                    self.activation_checkpointing_enabled
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    hidden_states = activation_checkpoint(
                        lambda hidden, current_block=block: current_block(
                            hidden,
                            attention_mask=attention_mask,
                            active_modules=per_example_modules,
                            active_experts=self.active_experts,
                            routing_granularity=self.routing_granularity,
                        ),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        active_modules=per_example_modules,
                        active_experts=self.active_experts,
                        routing_granularity=self.routing_granularity,
                    )
            return hidden_states

        def adapt_warm_start_state_dict(self, state_dict):
            """Translate dense/post-core layers into a routed-FFN warm start."""

            if self.mop_block_type != "routed_ffn":
                self.last_warm_start_metadata = {"adapted": False}
                return dict(state_dict)
            source_prefix = None
            for candidate in ("blocks.layers", "shared_blocks.layers"):
                if any(key.startswith(f"{candidate}.") for key in state_dict):
                    source_prefix = candidate
                    break
            if source_prefix is None:
                self.last_warm_start_metadata = {
                    "adapted": False,
                    "reason": "compatible_dense_layers_not_found",
                }
                return dict(state_dict)

            adapted = dict(state_dict)
            mapped_keys: set[str] = set()
            for layer_index in range(self.shared_layer_count):
                source = f"{source_prefix}.{layer_index}."
                target = f"shared_blocks.layers.{layer_index}."
                for name, tensor in state_dict.items():
                    if name.startswith(source):
                        target_name = target + name[len(source) :]
                        adapted[target_name] = tensor
                        mapped_keys.add(target_name)

            expert_clones = 0
            suffix_map = {
                "self_attn.in_proj_weight": "self_attn.in_proj_weight",
                "self_attn.in_proj_bias": "self_attn.in_proj_bias",
                "self_attn.out_proj.weight": "self_attn.out_proj.weight",
                "self_attn.out_proj.bias": "self_attn.out_proj.bias",
                "norm1.weight": "attn_norm.weight",
                "norm1.bias": "attn_norm.bias",
                "norm2.weight": "ffn_norm.weight",
                "norm2.bias": "ffn_norm.bias",
            }
            for routed_index in range(self.routed_layer_count):
                source_index = self.shared_layer_count + routed_index
                source = f"{source_prefix}.{source_index}."
                target = f"routed_blocks.{routed_index}."
                for source_suffix, target_suffix in suffix_map.items():
                    source_name = source + source_suffix
                    if source_name in state_dict:
                        target_name = target + target_suffix
                        adapted[target_name] = state_dict[source_name]
                        mapped_keys.add(target_name)
                for expert_name in self.routed_expert_names:
                    for source_suffix, expert_suffix in (
                        ("linear1.weight", "0.weight"),
                        ("linear1.bias", "0.bias"),
                        ("linear2.weight", "3.weight"),
                        ("linear2.bias", "3.bias"),
                    ):
                        source_name = source + source_suffix
                        if source_name in state_dict:
                            target_name = (
                                f"{target}experts.{expert_name}.{expert_suffix}"
                            )
                            adapted[target_name] = state_dict[source_name]
                            mapped_keys.add(target_name)
                    expert_clones += 1
            self.last_warm_start_metadata = {
                "adapted": bool(mapped_keys),
                "source_layer_prefix": source_prefix,
                "mapped_key_count": len(mapped_keys),
                "shared_layer_count": self.shared_layer_count,
                "routed_layer_count": self.routed_layer_count,
                "expert_clone_count": expert_clones,
            }
            return adapted

        def _apply_module_bank(self, shared_hidden, per_example_modules):
            routed_chunks = []
            for example_index, module_names in enumerate(per_example_modules):
                example_hidden = shared_hidden[example_index : example_index + 1]
                module_delta = self.module_bank.forward_one(
                    example_hidden, module_names
                )
                routed_chunks.append(example_hidden + module_delta)
            return torch.cat(routed_chunks, dim=0)

        def _loss_from_logits(self, logits, labels):
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                if (shift_labels != -100).any():
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    return loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                return logits.sum() * 0.0
            return None

        def _module_fully_frozen(self, *modules) -> bool:
            parameters = []
            for module in modules:
                if module is None:
                    continue
                parameters.extend(list(module.parameters()))
            return bool(parameters) and all(not parameter.requires_grad for parameter in parameters)

        def _module_param_count(self, *modules) -> int:
            total = 0
            for module in modules:
                if module is None:
                    continue
                total += sum(int(parameter.numel()) for parameter in module.parameters())
            return total

        def _expand_active_modules(
            self,
            active_modules: Sequence[str] | Sequence[Sequence[str]] | None,
            batch_size: int,
        ) -> list[list[str]]:
            if active_modules is None:
                return [
                    normalize_target_modules(
                        [],
                        self.module_names,
                        always_include_core=self.always_include_core,
                    )
                    for _ in range(batch_size)
                ]

            modules_list = list(active_modules)
            if not modules_list:
                return [
                    normalize_target_modules(
                        [],
                        self.module_names,
                        always_include_core=self.always_include_core,
                    )
                    for _ in range(batch_size)
                ]

            if all(isinstance(module, str) for module in modules_list):
                normalized = normalize_target_modules(
                    modules_list,
                    self.module_names,
                    always_include_core=self.always_include_core,
                )
                return [normalized for _ in range(batch_size)]

            if len(modules_list) != batch_size:
                raise ValueError(
                    "Per-example active_modules length must match batch size."
                )

            return [
                normalize_target_modules(
                    module_names,
                    self.module_names,
                    always_include_core=self.always_include_core,
                )
                for module_names in modules_list
            ]

        def _expand_active_adapters(
            self,
            active_adapters: Sequence[str] | Sequence[Sequence[str]] | None,
            batch_size: int,
        ) -> list[list[str]]:
            if active_adapters is None:
                return [[] for _ in range(batch_size)]

            adapters_list = (
                [active_adapters]
                if isinstance(active_adapters, str)
                else list(active_adapters)
            )
            if not adapters_list:
                return [[] for _ in range(batch_size)]

            if all(isinstance(adapter, str) for adapter in adapters_list):
                normalized = normalize_adapter_names(
                    adapters_list,
                    self.fast_adapter_names,
                )
                return [normalized for _ in range(batch_size)]

            if len(adapters_list) != batch_size:
                raise ValueError(
                    "Per-example active_adapters length must match batch size."
                )

            return [
                normalize_adapter_names(adapter_names, self.fast_adapter_names)
                for adapter_names in adapters_list
            ]

        def _expand_active_conditions(self, active_conditions, batch_size: int) -> list[list[str]]:
            if self.generated_adapter is None:
                return [[] for _ in range(batch_size)]
            return self.generated_adapter.expand_active_conditions(
                active_conditions,
                batch_size,
            )


    def _expert_names(module_names: Sequence[str], expert_count: int) -> list[str]:
        names = list(module_names)
        while len(names) < expert_count:
            names.append(f"expert_{len(names)}")
        return names[:expert_count]
