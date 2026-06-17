"""Tokenizer registry and factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mopforge.tokenization.base import TokenizerSpec
from mopforge.tokenization.byte_tokenizer import ByteTokenizer
from mopforge.tokenization.hf_tokenizer import HFTokenizerWrapper

TokenizerBuilder = Callable[[TokenizerSpec], Any]

_TOKENIZER_BUILDERS: dict[str, TokenizerBuilder] = {}


def register_tokenizer_type(name: str, builder: TokenizerBuilder) -> None:
    """Register a tokenizer builder for future extension."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")
    if not callable(builder):
        raise TypeError("builder must be callable.")
    _TOKENIZER_BUILDERS[name.strip().lower()] = builder


def build_tokenizer(spec: TokenizerSpec | dict[str, Any] | None = None):
    """Build a tokenizer from a spec.

    ``tokenizer_type="byte"`` returns the deterministic ``ByteTokenizer``.
    ``tokenizer_type="hf"`` returns ``HFTokenizerWrapper`` and may require
    optional dependencies and a local tokenizer path.
    """

    tokenizer_spec = _coerce_spec(spec)
    builder = _TOKENIZER_BUILDERS.get(tokenizer_spec.tokenizer_type)
    if builder is None:
        valid = ", ".join(sorted(_TOKENIZER_BUILDERS))
        raise ValueError(
            f"Unsupported tokenizer_type {tokenizer_spec.tokenizer_type!r}. "
            f"Known tokenizer types: {valid}."
        )
    return builder(tokenizer_spec)


def tokenizer_spec_from_config(config_or_dict: Any) -> TokenizerSpec:
    """Create a ``TokenizerSpec`` from config fields or a dictionary.

    If ``tokenizer_spec_path`` is present, it wins over inline fields.
    """

    data = _config_to_dict(config_or_dict)
    spec_path = data.get("tokenizer_spec_path")
    if spec_path:
        return TokenizerSpec.load_json(Path(spec_path))

    metadata = dict(data.get("tokenizer_metadata", {}) or {})
    return TokenizerSpec(
        tokenizer_type=data.get("tokenizer_type", "byte"),
        name_or_path=data.get("tokenizer_name_or_path")
        or data.get("name_or_path"),
        vocab_size=_first_present(data, "tokenizer_vocab_size", "vocab_size"),
        pad_token_id=_first_present(data, "tokenizer_pad_token_id", "pad_token_id"),
        bos_token_id=_first_present(data, "tokenizer_bos_token_id", "bos_token_id"),
        eos_token_id=_first_present(data, "tokenizer_eos_token_id", "eos_token_id"),
        unk_token_id=_first_present(data, "tokenizer_unk_token_id", "unk_token_id"),
        metadata=metadata,
    )


def _coerce_spec(spec: TokenizerSpec | dict[str, Any] | None) -> TokenizerSpec:
    if spec is None:
        return TokenizerSpec()
    if isinstance(spec, TokenizerSpec):
        return spec
    return TokenizerSpec.from_dict(spec)


def _config_to_dict(config_or_dict: Any) -> dict[str, Any]:
    if config_or_dict is None:
        return {}
    if isinstance(config_or_dict, dict):
        return dict(config_or_dict)
    if hasattr(config_or_dict, "to_dict"):
        return dict(config_or_dict.to_dict())
    return {
        name: getattr(config_or_dict, name)
        for name in dir(config_or_dict)
        if not name.startswith("_") and not callable(getattr(config_or_dict, name))
    }


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _build_byte_tokenizer(spec: TokenizerSpec) -> ByteTokenizer:
    return ByteTokenizer()


def _build_hf_tokenizer(spec: TokenizerSpec) -> HFTokenizerWrapper:
    if not spec.name_or_path:
        raise ValueError("HF tokenizer specs require name_or_path.")
    kwargs = dict(spec.metadata.get("kwargs", {}) or {})
    return HFTokenizerWrapper(spec.name_or_path, **kwargs)


register_tokenizer_type("byte", _build_byte_tokenizer)
register_tokenizer_type("hf", _build_hf_tokenizer)
