"""Demonstrate tokenizer specs and generic dataset compatibility."""

from __future__ import annotations

import os
from pathlib import Path

from mopforge.pretrain import CorpusCausalLMDataset, TextCorpusRecord
from mopforge.tokenization import (
    HFTokenizerWrapper,
    TokenizerSpec,
    build_tokenizer,
    get_tokenizer_vocab_size,
)


def main() -> None:
    print(
        "Tokenizer abstraction demo only. HF tokenizer support is optional and "
        "not required for CPU smoke tests."
    )

    output_dir = Path("outputs/tokenizer_abstraction_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TokenizerSpec(tokenizer_type="byte")
    tokenizer = build_tokenizer(spec)
    realized_spec = tokenizer.to_spec() if hasattr(tokenizer, "to_spec") else spec

    snippet = "def add(a, b):\n    return a + b\n"
    token_ids = tokenizer.encode(snippet)
    decoded = tokenizer.decode(token_ids)

    spec_path = realized_spec.save_json(output_dir / "tokenizer_spec.json")
    loaded_spec = TokenizerSpec.load_json(spec_path)

    dataset = CorpusCausalLMDataset(
        [
            TextCorpusRecord(
                id="demo-snippet",
                text=snippet,
                source="tokenizer_abstraction_demo",
                domain="coding",
                language="python",
            )
        ],
        tokenizer,
        max_seq_len=16,
        stride=8,
    )

    print(f"tokenizer_type={loaded_spec.tokenizer_type}")
    print(f"vocab_size={get_tokenizer_vocab_size(tokenizer)}")
    print(f"pad_token_id={tokenizer.pad_token_id}")
    print(f"bos_token_id={tokenizer.bos_token_id}")
    print(f"eos_token_id={tokenizer.eos_token_id}")
    print(f"encoded_tokens={len(token_ids)}")
    print(f"decoded_matches={decoded == snippet}")
    print(f"dataset_chunks={len(dataset)}")
    print(f"tokenizer_spec_json={spec_path}")

    hf_path = os.environ.get("MOPFORGE_HF_TOKENIZER_PATH")
    if hf_path:
        try:
            hf_tokenizer = HFTokenizerWrapper(hf_path, local_files_only=True)
        except Exception as exc:
            print(f"hf_optional_demo=skipped ({exc.__class__.__name__}: {exc})")
        else:
            print(f"hf_optional_demo=loaded vocab_size={get_tokenizer_vocab_size(hf_tokenizer)}")
    else:
        print("hf_optional_demo=skipped (set MOPFORGE_HF_TOKENIZER_PATH for a local tokenizer)")


if __name__ == "__main__":
    main()
