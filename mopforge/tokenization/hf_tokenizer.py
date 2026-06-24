"""Optional Hugging Face/tokenizers compatibility wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mopforge.tokenization.base import TokenizerSpec, get_tokenizer_vocab_size


class HFTokenizerWrapper:
    """Thin wrapper around a local Hugging Face or ``tokenizers`` tokenizer.

    Dependencies are imported only when this wrapper is instantiated. By
    default, ``transformers.AutoTokenizer`` is loaded with
    ``local_files_only=True`` so tests and CPU smoke examples never require
    internet access.
    """

    def __init__(self, name_or_path: str, **kwargs: Any) -> None:
        """Load a tokenizer from ``name_or_path``."""

        if not isinstance(name_or_path, str) or not name_or_path.strip():
            raise ValueError("name_or_path must be a non-empty string.")
        self.name_or_path = name_or_path.strip()
        self.kwargs = dict(kwargs)
        self._backend = ""
        self._tokenizer = self._load_tokenizer()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text using the wrapped tokenizer."""

        if self._backend == "transformers":
            return list(
                self._tokenizer.encode(
                    text,
                    add_special_tokens=add_special_tokens,
                )
            )
        return list(
            self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            ).ids
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs using the wrapped tokenizer."""

        return str(
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
            )
        )

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size for the wrapped tokenizer."""

        if self._backend == "transformers":
            return int(len(self._tokenizer))
        return int(self._tokenizer.get_vocab_size())

    @property
    def pad_token_id(self) -> int | None:
        """Return pad token ID if the wrapped tokenizer has one."""

        return self._special_token_id("pad_token_id", "<pad>")

    @property
    def bos_token_id(self) -> int | None:
        """Return BOS token ID if the wrapped tokenizer has one."""

        return self._special_token_id("bos_token_id", "<bos>")

    @property
    def eos_token_id(self) -> int | None:
        """Return EOS token ID if the wrapped tokenizer has one."""

        return self._special_token_id("eos_token_id", "<eos>")

    @property
    def unk_token_id(self) -> int | None:
        """Return unknown token ID if the wrapped tokenizer has one."""

        return self._special_token_id("unk_token_id", "<unk>")

    def to_spec(self) -> TokenizerSpec:
        """Return a serializable spec for this tokenizer."""

        return TokenizerSpec(
            tokenizer_type="hf",
            name_or_path=self.name_or_path,
            vocab_size=get_tokenizer_vocab_size(self),
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id,
            metadata={"backend": self._backend},
        )

    def _load_tokenizer(self):
        kwargs = dict(self.kwargs)
        local_files_only = bool(kwargs.pop("local_files_only", True))
        path = Path(self.name_or_path)
        if path.is_file() and path.suffix.lower() == ".json":
            try:
                from tokenizers import Tokenizer
            except ImportError as tokenizers_exc:
                raise ImportError(_install_hint()) from tokenizers_exc
            self._backend = "tokenizers"
            return Tokenizer.from_file(str(path))
        try:
            from transformers import AutoTokenizer
        except ImportError as transformers_exc:
            if path.exists():
                try:
                    from tokenizers import Tokenizer
                except ImportError as tokenizers_exc:
                    raise ImportError(_install_hint()) from tokenizers_exc
                self._backend = "tokenizers"
                return Tokenizer.from_file(str(path))
            raise ImportError(_install_hint()) from transformers_exc

        self._backend = "transformers"
        return AutoTokenizer.from_pretrained(
            self.name_or_path,
            local_files_only=local_files_only,
            **kwargs,
        )

    def _special_token_id(self, attr_name: str, token_text: str) -> int | None:
        if self._backend == "transformers":
            value = getattr(self._tokenizer, attr_name, None)
            return None if value is None else int(value)
        value = self._tokenizer.token_to_id(token_text)
        return None if value is None else int(value)


def _install_hint() -> str:
    return (
        "HF tokenizer support requires an optional dependency. Install "
        "`transformers` for AutoTokenizer support or `tokenizers` for local "
        "tokenizer JSON files, for example: pip install transformers"
    )
