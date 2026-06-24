import json

import pytest

from mopforge.gpu import (
    PackedTokenDataset,
    TokenShardBuildConfig,
    build_packed_token_dataloaders,
    build_token_shards,
)
from mopforge.tokenization import (
    BPETrainingConfig,
    TokenizerSpec,
    build_tokenizer,
    train_bpe_tokenizer,
)


pytest.importorskip("torch")
pytest.importorskip("tokenizers")


def _corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": f"doc-{index:04d}",
            "text": f"def function_{index}(value):\n    return value + {index}\n",
        }
        for index in range(200)
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_train_local_bpe_and_pack_memory_mapped_token_shards(tmp_path):
    corpus = _corpus(tmp_path)
    tokenizer_result = train_bpe_tokenizer(
        BPETrainingConfig(
            source_paths=[str(corpus)],
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=300,
            min_frequency=1,
        )
    )
    spec = TokenizerSpec.load_json(tokenizer_result["tokenizer_spec_path"])
    tokenizer = build_tokenizer(spec)

    assert tokenizer.vocab_size >= 260
    assert "function" in tokenizer.decode(tokenizer.encode("function", add_special_tokens=False))

    result = build_token_shards(
        TokenShardBuildConfig(
            source_paths=[str(corpus)],
            tokenizer_spec_path=tokenizer_result["tokenizer_spec_path"],
            output_dir=str(tmp_path / "packed"),
            sequence_length=16,
            tokens_per_shard=64,
            eval_fraction=0.2,
            split_seed=7,
        )
    )

    train = PackedTokenDataset(result["manifest_path"], "train")
    evaluation = PackedTokenDataset(result["manifest_path"], "eval")
    assert len(train) > 0
    assert len(evaluation) > 0
    assert train[0]["input_ids"].shape == (16,)
    assert train[0]["labels"].tolist() == train[0]["input_ids"].tolist()
    assert len(result["splits"]["train"]) > 1

    train_loader, eval_loader, metadata = build_packed_token_dataloaders(
        result["manifest_path"],
        micro_batch_size=2,
        num_workers=0,
        pin_memory=False,
        shuffle_train=True,
        shuffle_seed=42,
    )
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (2, 16)
    assert len(eval_loader.dataset) == len(evaluation)
    assert metadata["kind"] == "packed_token_shards"
    assert metadata["packing_efficiency"] <= 1.0
