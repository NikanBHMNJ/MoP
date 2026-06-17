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


    class TinyMoPCausalTransformer(nn.Module):
        """Tiny oracle-routed MoP causal LM for CPU smoke tests.

        Routing is not learned in Goal 4. The caller supplies active modules,
        usually from ``KnowledgeLesson.target_modules``. ``core`` is always
        included when it exists in ``module_names``.
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

            self.max_seq_len = max_seq_len
            self.use_fast_adapters = bool(use_fast_adapters)
            self.fast_adapter_names = list(fast_adapter_names or ["default"])
            self.use_generated_params = bool(use_generated_params)
            self.generated_condition_names = list(generated_condition_names or ["default"])
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.shared_blocks = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )
            self.module_bank = ModuleBank(self.module_names, d_model, dropout)
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

            device = input_ids.device
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            hidden_states = self.token_embedding(input_ids) + self.position_embedding(
                positions
            )
            hidden_states = self.dropout(hidden_states)

            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            padding_mask = attention_mask == 0 if attention_mask is not None else None
            shared_hidden = self.shared_blocks(
                hidden_states,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )

            per_example_modules = self._expand_active_modules(
                active_modules, batch_size
            )
            routed_chunks = []
            for example_index, module_names in enumerate(per_example_modules):
                example_hidden = shared_hidden[example_index : example_index + 1]
                module_delta = self.module_bank.forward_one(
                    example_hidden, module_names
                )
                routed_chunks.append(example_hidden + module_delta)
            routed_hidden = torch.cat(routed_chunks, dim=0)

            per_example_adapters = self._expand_active_adapters(
                active_adapters,
                batch_size,
            )
            if self.fast_adapter_bank is not None:
                routed_hidden = self.fast_adapter_bank(
                    routed_hidden,
                    active_adapters=per_example_adapters,
                )
            per_example_conditions = self._expand_active_conditions(
                active_conditions,
                batch_size,
            )
            if self.generated_adapter is not None:
                routed_hidden = self.generated_adapter(
                    routed_hidden,
                    active_conditions=per_example_conditions,
                )

            logits = self.lm_head(self.norm(routed_hidden))
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
                "active_modules": per_example_modules,
                "active_adapters": per_example_adapters,
                "active_conditions": per_example_conditions,
            }

        def _expand_active_modules(
            self,
            active_modules: Sequence[str] | Sequence[Sequence[str]] | None,
            batch_size: int,
        ) -> list[list[str]]:
            if active_modules is None:
                return [
                    normalize_target_modules([], self.module_names)
                    for _ in range(batch_size)
                ]

            modules_list = list(active_modules)
            if not modules_list:
                return [
                    normalize_target_modules([], self.module_names)
                    for _ in range(batch_size)
                ]

            if all(isinstance(module, str) for module in modules_list):
                normalized = normalize_target_modules(modules_list, self.module_names)
                return [normalized for _ in range(batch_size)]

            if len(modules_list) != batch_size:
                raise ValueError(
                    "Per-example active_modules length must match batch size."
                )

            return [
                normalize_target_modules(module_names, self.module_names)
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
