"""A deterministic UTF-8 byte-level tokenizer."""

from __future__ import annotations

from mopforge.tokenization.base import BaseTokenizer, TokenizerSpec


class ByteTokenizer(BaseTokenizer):
    """Simple byte-level tokenizer with fixed special token IDs.

    This tokenizer needs no training. Normal UTF-8 bytes map to stable IDs
    after three reserved special tokens:

    - ``<pad>`` -> 0
    - ``<bos>`` -> 1
    - ``<eos>`` -> 2
    - byte ``b`` -> ``b + 3``
    """

    _PAD_TOKEN_ID = 0
    _BOS_TOKEN_ID = 1
    _EOS_TOKEN_ID = 2
    _BYTE_OFFSET = 3
    _VOCAB_SIZE = 259

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode ``text`` into deterministic byte token IDs."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        token_ids = [byte + self._BYTE_OFFSET for byte in text.encode("utf-8")]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode byte token IDs into text.

        Invalid byte IDs are replaced with the Unicode replacement character.
        Special tokens are skipped by default.
        """

        pieces: list[str] = []
        byte_buffer = bytearray()
        for token_id in ids:
            if token_id in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }:
                if byte_buffer:
                    pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                if not skip_special_tokens:
                    pieces.append(self._special_token_text(token_id))
                continue

            byte_value = token_id - self._BYTE_OFFSET
            if 0 <= byte_value <= 255:
                byte_buffer.append(byte_value)
            else:
                if byte_buffer:
                    pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                pieces.append("\ufffd")

        if byte_buffer:
            pieces.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        """Return the fixed byte-token vocabulary size."""

        return self._VOCAB_SIZE

    @property
    def pad_token_id(self) -> int:
        """Return the ``<pad>`` token ID."""

        return self._PAD_TOKEN_ID

    @property
    def bos_token_id(self) -> int:
        """Return the ``<bos>`` token ID."""

        return self._BOS_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        """Return the ``<eos>`` token ID."""

        return self._EOS_TOKEN_ID

    @property
    def unk_token_id(self) -> int | None:
        """Return None because every UTF-8 byte is representable."""

        return None

    def to_spec(self) -> TokenizerSpec:
        """Return a serializable spec for this deterministic tokenizer."""

        return TokenizerSpec(
            tokenizer_type="byte",
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id,
            metadata={"encoding": "utf-8-byte"},
        )

    def _special_token_text(self, token_id: int) -> str:
        if token_id == self.pad_token_id:
            return "<pad>"
        if token_id == self.bos_token_id:
            return "<bos>"
        if token_id == self.eos_token_id:
            return "<eos>"
        raise ValueError(f"Unknown special token id: {token_id}")
