"""Base tokenizer interface and specs used by MoP-Forge data pipelines."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


SUPPORTED_TOKENIZER_TYPES = {"byte", "hf"}


@dataclass(slots=True)
class TokenizerSpec:
    """Serializable tokenizer configuration for data and run artifacts."""

    tokenizer_type: str = "byte"
    name_or_path: str | None = None
    vocab_size: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    unk_token_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize tokenizer metadata."""

        if not isinstance(self.tokenizer_type, str) or not self.tokenizer_type.strip():
            raise ValueError("tokenizer_type must be a non-empty string.")
        self.tokenizer_type = self.tokenizer_type.strip().lower()

        if self.name_or_path is not None:
            if not isinstance(self.name_or_path, str) or not self.name_or_path.strip():
                raise ValueError("name_or_path must be a non-empty string or None.")
            self.name_or_path = self.name_or_path.strip()

        if self.tokenizer_type == "hf" and self.name_or_path is None:
            raise ValueError("HF tokenizers require name_or_path.")

        if self.vocab_size is not None:
            if type(self.vocab_size) is not int or self.vocab_size <= 0:
                raise ValueError("vocab_size must be a positive integer or None.")

        for field_name in (
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None.")

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        try:
            json.dumps(self.metadata)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable.") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerSpec":
        """Create a spec from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("TokenizerSpec.from_dict expects a dictionary.")
        return cls(
            tokenizer_type=data.get("tokenizer_type", "byte"),
            name_or_path=data.get("name_or_path"),
            vocab_size=data.get("vocab_size"),
            pad_token_id=data.get("pad_token_id"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            unk_token_id=data.get("unk_token_id"),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def save_json(self, path: str | Path) -> Path:
        """Write this spec to JSON and return the path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> "TokenizerSpec":
        """Load a tokenizer spec from JSON."""

        input_path = Path(path)
        return cls.from_dict(json.loads(input_path.read_text(encoding="utf-8")))


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Structural tokenizer interface used by datasets and tiny trainers."""

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID, or None when unavailable."""

    @property
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token ID, or None."""

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID, or None."""

    @property
    def unk_token_id(self) -> int | None:
        """Return the unknown-token ID, or None."""

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size as a property.

        Registry helpers also accept tokenizers that expose ``vocab_size()`` as
        a method for compatibility with external wrappers.
        """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text into token IDs."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into text."""


def get_tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return vocab size from a tokenizer property or method."""

    value = getattr(tokenizer, "vocab_size", None)
    if value is None:
        raise AttributeError("tokenizer must expose vocab_size.")
    if callable(value):
        value = value()
    if type(value) is not int or value <= 0:
        raise ValueError("tokenizer vocab_size must be a positive integer.")
    return value


def get_tokenizer_special_token_id(tokenizer: Any, field_name: str) -> int | None:
    """Return a special token ID from ``tokenizer`` if it exists."""

    value = getattr(tokenizer, field_name, None)
    if callable(value):
        value = value()
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"tokenizer {field_name} must be a non-negative integer or None.")
    return value


def get_tokenizer_pad_token_id(tokenizer: Any, default: int = 0) -> int:
    """Return tokenizer pad ID, falling back to ``default`` when absent."""

    value = get_tokenizer_special_token_id(tokenizer, "pad_token_id")
    return default if value is None else value


class BaseTokenizer(ABC):
    """Small tokenizer interface for causal-LM data preparation."""

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text into token IDs."""

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back into text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the number of token IDs in the vocabulary."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Return the padding token ID."""

    @property
    @abstractmethod
    def bos_token_id(self) -> int | None:
        """Return the beginning-of-sequence token ID."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID."""

    @property
    @abstractmethod
    def unk_token_id(self) -> int | None:
        """Return the unknown-token ID, or None when unsupported."""
