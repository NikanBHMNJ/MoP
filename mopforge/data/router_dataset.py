"""Dataset and collator for learned module routing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mopforge.kts import KnowledgeLesson
from mopforge.tokenization import (
    ByteTokenizer,
    TokenizerProtocol,
    get_tokenizer_pad_token_id,
    get_tokenizer_special_token_id,
)
from mopforge.training import DEFAULT_KNOWN_MODULES, module_mask_from_targets


class RouterDataset:
    """Tokenized router examples from ``KnowledgeLesson`` records."""

    def __init__(
        self,
        lessons: list[KnowledgeLesson],
        tokenizer: TokenizerProtocol | None = None,
        *,
        known_modules: list[str] | None = None,
        max_length: int = 512,
        strict_modules: bool = False,
    ) -> None:
        """Create deterministic learned-router examples.

        The router input uses task metadata and lesson input only. It does not
        include ``expected_output`` by default.
        """

        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        self.lessons = list(lessons)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.known_modules = list(known_modules or DEFAULT_KNOWN_MODULES)
        self.max_length = max_length
        self.strict_modules = strict_modules
        for lesson in self.lessons:
            lesson.validate()

    def __len__(self) -> int:
        """Return the number of routing examples."""

        return len(self.lessons)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one tokenized router example."""

        lesson = self.lessons[idx]
        text = format_lesson_for_router(lesson)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            eos_token_id = get_tokenizer_special_token_id(self.tokenizer, "eos_token_id")
            if eos_token_id is not None:
                input_ids[-1] = eos_token_id

        normalized_mask = module_mask_from_targets(
            lesson.target_modules,
            self.known_modules,
            strict=self.strict_modules,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "module_mask": normalized_mask,
            "target_modules": list(lesson.target_modules),
            "metadata": deepcopy(lesson.metadata),
            "lesson_id": lesson.id,
            "route_text": text,
        }


def format_lesson_for_router(lesson: KnowledgeLesson) -> str:
    """Create deterministic task text for the learned router."""

    lesson.validate()
    subskill = lesson.subskill or "none"
    concept = lesson.concept or "none"
    return (
        "Route this lesson to MoP modules.\n"
        f"Domain: {lesson.domain}\n"
        f"Skill: {lesson.skill}\n"
        f"Subskill: {subskill}\n"
        f"Difficulty: {lesson.difficulty}\n"
        f"Concept: {concept}\n"
        "Input:\n"
        f"{lesson.input.rstrip()}\n"
    )


try:
    import torch
except Exception:
    torch = None
    RouterCollator = None
else:

    class RouterCollator:
        """Pad router examples and stack multi-label module masks."""

        def __init__(self, tokenizer: TokenizerProtocol | None = None) -> None:
            self.tokenizer = tokenizer or ByteTokenizer()

        def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
            if not batch:
                raise ValueError("batch must contain at least one item.")

            max_length = max(len(item["input_ids"]) for item in batch)
            input_ids: list[list[int]] = []
            attention_mask: list[list[int]] = []
            module_mask: list[list[int]] = []
            pad_token_id = get_tokenizer_pad_token_id(self.tokenizer)

            for item in batch:
                pad_length = max_length - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [pad_token_id] * pad_length)
                attention_mask.append(item["attention_mask"] + [0] * pad_length)
                module_mask.append(item["module_mask"])

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "module_mask": torch.tensor(module_mask, dtype=torch.float32),
                "target_modules": [item["target_modules"] for item in batch],
                "metadata": [item.get("metadata", {}) for item in batch],
                "lesson_id": [item["lesson_id"] for item in batch],
                "route_text": [item["route_text"] for item in batch],
            }
