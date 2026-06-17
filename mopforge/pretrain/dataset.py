"""Causal-LM dataset for continued-pretraining text corpora."""

from __future__ import annotations

from typing import Any

from mopforge.pretrain.corpus import TextCorpusRecord
from mopforge.tokenization import (
    TokenizerProtocol,
    get_tokenizer_pad_token_id,
    get_tokenizer_special_token_id,
)


class CorpusCausalLMDataset:
    """Tokenized full-sequence causal-LM chunks from text corpus records."""

    def __init__(
        self,
        records: list[TextCorpusRecord],
        tokenizer: TokenizerProtocol,
        max_seq_len: int = 512,
        stride: int | None = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        """Create deterministic token chunks for corpus pretraining."""

        if type(max_seq_len) is not int or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if stride is not None and (type(stride) is not int or stride <= 0):
            raise ValueError("stride must be a positive integer or None.")
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self._items = self._build_items()

    def __len__(self) -> int:
        """Return chunk count."""

        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one full-sequence causal-LM chunk."""

        item = self._items[index]
        token_ids = list(item["input_ids"])
        return {
            "input_ids": token_ids,
            "labels": list(token_ids),
            "attention_mask": [1] * len(token_ids),
            "record_id": item["record_id"],
            "chunk_index": item["chunk_index"],
        }

    def _build_items(self) -> list[dict[str, Any]]:
        items = []
        for record in self.records:
            record.validate()
            token_ids = self.tokenizer.encode(record.text, add_special_tokens=False)
            bos_token_id = get_tokenizer_special_token_id(self.tokenizer, "bos_token_id")
            eos_token_id = get_tokenizer_special_token_id(self.tokenizer, "eos_token_id")
            if self.add_bos and bos_token_id is not None:
                token_ids = [bos_token_id, *token_ids]
            if self.add_eos and eos_token_id is not None:
                token_ids = [*token_ids, eos_token_id]
            chunk_index = 0
            for start in range(0, len(token_ids), self.stride):
                chunk = token_ids[start : start + self.max_seq_len]
                if not chunk:
                    continue
                items.append(
                    {
                        "input_ids": chunk,
                        "record_id": record.id,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
        return items


try:
    import torch
except Exception:
    torch = None
    CorpusCausalLMCollator = None
else:

    class CorpusCausalLMCollator:
        """Pad corpus chunks into PyTorch tensors."""

        def __init__(self, tokenizer: TokenizerProtocol) -> None:
            self.tokenizer = tokenizer

        def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
            if not batch:
                raise ValueError("batch must contain at least one item.")
            max_length = max(len(item["input_ids"]) for item in batch)
            input_ids = []
            labels = []
            attention_mask = []
            pad_token_id = get_tokenizer_pad_token_id(self.tokenizer)
            for item in batch:
                pad_length = max_length - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad_token_id] * pad_length)
                labels.append(item["labels"] + [-100] * pad_length)
                attention_mask.append(item["attention_mask"] + [0] * pad_length)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "record_id": [item["record_id"] for item in batch],
                "chunk_index": [item["chunk_index"] for item in batch],
            }
