"""Tests for MoP-Forge tokenizers."""

from __future__ import annotations

from mopforge.tokenization import ByteTokenizer


def test_byte_tokenizer_round_trip_normal_and_unicode_text() -> None:
    tokenizer = ByteTokenizer()
    text = "Fix return value: total + shipping\nUnicode: سلام"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_byte_tokenizer_special_token_ids_and_vocab_size() -> None:
    tokenizer = ByteTokenizer()

    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.unk_token_id is None
    assert tokenizer.vocab_size == 259
    assert tokenizer.encode("A") == [1, 68, 2]
    assert tokenizer.decode([1, 68, 2], skip_special_tokens=False) == "<bos>A<eos>"
