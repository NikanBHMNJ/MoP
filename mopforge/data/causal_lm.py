"""Causal-LM dataset and collator for KnowledgeLesson records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mopforge.formatting import format_lesson_for_causal_lm
from mopforge.kts import KnowledgeLesson
from mopforge.tokenization import (
    TokenizerProtocol,
    get_tokenizer_pad_token_id,
    get_tokenizer_special_token_id,
)


class LessonCausalLMDataset:
    """Tokenized causal-LM dataset over structured lessons."""

    def __init__(
        self,
        lessons: list[KnowledgeLesson],
        tokenizer: TokenizerProtocol,
        max_length: int = 2048,
        mask_prompt_labels: bool = True,
    ) -> None:
        """Create a model-ready dataset.

        Args:
            lessons: Valid KTS lessons.
            tokenizer: Tokenizer implementing the MoP-Forge tokenizer interface.
            max_length: Maximum sequence length after right truncation.
            mask_prompt_labels: If True, prompt token labels are set to -100.
        """

        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        self.lessons = list(lessons)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt_labels = mask_prompt_labels
        for lesson in self.lessons:
            lesson.validate()

    def __len__(self) -> int:
        """Return the number of lessons."""

        return len(self.lessons)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one tokenized causal-LM training item."""

        lesson = self.lessons[idx]
        formatted = format_lesson_for_causal_lm(lesson)
        prompt = str(formatted["prompt"])
        target = str(formatted["target"])

        bos_token_id = get_tokenizer_special_token_id(self.tokenizer, "bos_token_id")
        eos_token_id = get_tokenizer_special_token_id(self.tokenizer, "eos_token_id")

        prompt_ids = []
        if bos_token_id is not None:
            prompt_ids.append(bos_token_id)
        prompt_ids.extend(self.tokenizer.encode(prompt, add_special_tokens=False))

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if eos_token_id is not None:
            target_ids.append(eos_token_id)

        input_ids = prompt_ids + target_ids
        if self.mask_prompt_labels:
            labels = [-100] * len(prompt_ids) + list(target_ids)
        else:
            labels = list(input_ids)
        attention_mask = [1] * len(input_ids)

        input_ids, labels, attention_mask = self._truncate(
            input_ids, labels, attention_mask, len(prompt_ids)
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "lesson_id": lesson.id,
            "target_modules": list(lesson.target_modules),
            "metadata": deepcopy(lesson.metadata),
            "domain": lesson.domain,
            "skill": lesson.skill,
        }

    def sequence_length_statistics(self) -> dict[str, Any]:
        """Measure prompt/target lengths before right truncation."""

        bos_count = int(
            get_tokenizer_special_token_id(self.tokenizer, "bos_token_id") is not None
        )
        eos_count = int(
            get_tokenizer_special_token_id(self.tokenizer, "eos_token_id") is not None
        )
        prompt_lengths: list[int] = []
        target_lengths: list[int] = []
        sequence_lengths: list[int] = []
        for lesson in self.lessons:
            formatted = format_lesson_for_causal_lm(lesson)
            prompt_length = bos_count + len(
                self.tokenizer.encode(str(formatted["prompt"]), add_special_tokens=False)
            )
            target_length = len(
                self.tokenizer.encode(str(formatted["target"]), add_special_tokens=False)
            ) + eos_count
            prompt_lengths.append(prompt_length)
            target_lengths.append(target_length)
            sequence_lengths.append(prompt_length + target_length)
        truncated = sum(length > self.max_length for length in sequence_lengths)
        return {
            "examples": len(sequence_lengths),
            "max_seq_len": self.max_length,
            "truncated_examples": truncated,
            "truncated_rate": truncated / len(sequence_lengths) if sequence_lengths else 0.0,
            "max_original_sequence_tokens": max(sequence_lengths, default=0),
            "max_prompt_tokens": max(prompt_lengths, default=0),
            "max_target_tokens": max(target_lengths, default=0),
        }

    def _truncate(
        self,
        input_ids: list[int],
        labels: list[int],
        attention_mask: list[int],
        prompt_length: int,
    ) -> tuple[list[int], list[int], list[int]]:
        if len(input_ids) <= self.max_length:
            return input_ids, labels, attention_mask

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = attention_mask[: self.max_length]

        # Keep an EOS marker when right truncation removed the original one.
        eos_token_id = get_tokenizer_special_token_id(self.tokenizer, "eos_token_id")
        if input_ids and eos_token_id is not None:
            input_ids[-1] = eos_token_id
            if not self.mask_prompt_labels or self.max_length > prompt_length:
                labels[-1] = eos_token_id
        return input_ids, labels, attention_mask


try:
    import torch
except Exception:
    torch = None
    CausalLMCollator = None
else:

    class CausalLMCollator:
        """Pad causal-LM items into PyTorch tensors."""

        def __init__(self, tokenizer: TokenizerProtocol) -> None:
            self.tokenizer = tokenizer

        def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
            if not batch:
                raise ValueError("batch must contain at least one item.")

            max_length = max(len(item["input_ids"]) for item in batch)
            input_ids: list[list[int]] = []
            labels: list[list[int]] = []
            attention_mask: list[list[int]] = []
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
                "lesson_id": [item["lesson_id"] for item in batch],
                "target_modules": [item["target_modules"] for item in batch],
                "metadata": [item.get("metadata", {}) for item in batch],
                "domain": [item["domain"] for item in batch],
                "skill": [item["skill"] for item in batch],
            }
