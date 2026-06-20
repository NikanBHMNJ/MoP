"""Optional tiny dense causal transformer baseline."""

from __future__ import annotations

from collections.abc import Sequence

from mopforge.models.fast_adapters import (
    FastAdapterBank,
    FastAdapterConfig,
    normalize_adapter_names,
)
from mopforge.models.generated_params import GeneratedAdapter, GeneratedParameterConfig


try:
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except Exception:
    torch = None
    nn = None
    TinyCausalTransformer = None
else:

    class TinyCausalTransformer(nn.Module):
        """A tiny causal transformer for data-pipeline smoke tests."""

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            max_seq_len: int = 512,
            dropout: float = 0.0,
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

            self.max_seq_len = max_seq_len
            self.use_fast_adapters = bool(use_fast_adapters)
            self.fast_adapter_names = list(fast_adapter_names or ["default"])
            self.use_generated_params = bool(use_generated_params)
            self.generated_condition_names = list(generated_condition_names or ["default"])
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation_checkpointing_enabled = False
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
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
            active_adapters=None,
            active_conditions=None,
        ):
            """Run a causal-LM forward pass and optional next-token loss."""

            batch_size, seq_len = input_ids.shape
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_seq_len "
                    f"{self.max_seq_len}."
                )

            device = input_ids.device
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = self.dropout(x)

            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            padding_mask = None
            if attention_mask is not None:
                padding_mask = attention_mask == 0

            if (
                self.activation_checkpointing_enabled
                and self.training
                and torch.is_grad_enabled()
            ):
                for layer in self.blocks.layers:
                    x = activation_checkpoint(
                        lambda hidden, current_layer=layer: current_layer(
                            hidden,
                            src_mask=causal_mask,
                            src_key_padding_mask=padding_mask,
                            is_causal=True,
                        ),
                        x,
                        use_reentrant=False,
                    )
            else:
                x = self.blocks(
                    x,
                    mask=causal_mask,
                    src_key_padding_mask=padding_mask,
                    is_causal=True,
                )
            per_example_adapters = self._expand_active_adapters(
                active_adapters,
                batch_size,
            )
            if self.fast_adapter_bank is not None:
                x = self.fast_adapter_bank(x, active_adapters=per_example_adapters)
            per_example_conditions = self._expand_active_conditions(
                active_conditions,
                batch_size,
            )
            if self.generated_adapter is not None:
                x = self.generated_adapter(
                    x,
                    active_conditions=per_example_conditions,
                )
            x = self.norm(x)
            logits = self.lm_head(x)

            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                if (shift_labels != -100).any():
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                else:
                    loss = logits.sum() * 0.0

            return {
                "logits": logits,
                "loss": loss,
                "active_adapters": per_example_adapters,
                "active_conditions": per_example_conditions,
            }

        def _expand_active_adapters(self, active_adapters, batch_size: int) -> list[list[str]]:
            if active_adapters is None:
                return [[] for _ in range(batch_size)]

            adapters_list = list(active_adapters) if not isinstance(active_adapters, str) else [active_adapters]
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
