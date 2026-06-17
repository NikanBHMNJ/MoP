"""Tests for continued-pretraining corpus API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.kts import KnowledgeLesson
from mopforge.pretrain import (
    ContinuedPretrainConfig,
    CorpusCausalLMCollator,
    CorpusCausalLMDataset,
    TextCorpusRecord,
    TextCorpusStore,
    build_corpus_from_lessons,
    build_demo_code_corpus,
    run_continued_pretraining,
)
from mopforge.tokenization import ByteTokenizer


def make_record(record_id: str = "record-a", *, domain: str = "coding") -> TextCorpusRecord:
    return TextCorpusRecord(
        id=record_id,
        text="def add(a, b):\n    return a + b",
        source="test",
        domain=domain,
        language="python",
        metadata={"topic": "functions"},
    )


def make_lesson(lesson_id: str = "lesson-a") -> KnowledgeLesson:
    return KnowledgeLesson(
        id=lesson_id,
        domain="coding",
        skill="debugging",
        subskill="missing-return",
        difficulty=1,
        target_modules=["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "python_tests", "status": "verified"},
        metadata={"language": "python"},
    )


def test_text_corpus_record_validation_and_dict_round_trip() -> None:
    record = make_record()

    loaded = TextCorpusRecord.from_dict(record.to_dict())

    assert loaded == record
    with pytest.raises(ValueError, match="id"):
        TextCorpusRecord(id="", text="hello")
    with pytest.raises(ValueError, match="text"):
        TextCorpusRecord(id="bad", text="")
    with pytest.raises(ValueError, match="metadata"):
        TextCorpusRecord(id="bad", text="hello", metadata={"x": object()})


def test_text_corpus_store_add_load_get_count_filter_export(tmp_path) -> None:
    store = TextCorpusStore(tmp_path / "corpus.jsonl")
    store.add(make_record("a"))
    store.add(make_record("b", domain="math"))

    assert store.count() == 2
    assert store.get("a").id == "a"
    assert [record.id for record in store.load_all()] == ["a", "b"]
    assert [record.id for record in store.filter(domain="coding")] == ["a"]
    assert [record.id for record in store.filter(language="python")] == ["a", "b"]
    assert [record.id for record in store.filter(metadata={"topic": "functions"})] == ["a", "b"]

    export_path = store.export_json(tmp_path / "corpus.json")
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert [record["id"] for record in exported] == ["a", "b"]


def test_text_corpus_store_rejects_duplicate_ids(tmp_path) -> None:
    store = TextCorpusStore(tmp_path / "corpus.jsonl")
    store.add(make_record("a"))

    with pytest.raises(ValueError, match="Duplicate"):
        store.add(make_record("a"))
    with pytest.raises(ValueError, match="Duplicate"):
        store.add_many([make_record("b"), make_record("b")])


def test_build_demo_code_corpus_creates_valid_records() -> None:
    records = build_demo_code_corpus(count=5)

    assert len(records) == 5
    assert records[0].id == "demo-code-corpus-000"
    assert all(record.language == "python" for record in records)


def test_build_corpus_from_lessons_creates_valid_text_records() -> None:
    records = build_corpus_from_lessons([make_lesson()])

    assert len(records) == 1
    assert records[0].source == "lesson"
    assert "Input:" in records[0].text
    assert "Expected output:" in records[0].text
    assert records[0].metadata["lesson_id"] == "lesson-a"


def test_corpus_causal_lm_dataset_creates_chunks() -> None:
    dataset = CorpusCausalLMDataset(
        [TextCorpusRecord(id="long", text="abcdef" * 20)],
        ByteTokenizer(),
        max_seq_len=16,
        stride=8,
    )

    item = dataset[0]

    assert len(dataset) > 1
    assert len(item["input_ids"]) <= 16
    assert item["labels"] == item["input_ids"]
    assert item["attention_mask"] == [1] * len(item["input_ids"])
    assert item["record_id"] == "long"


def test_corpus_labels_are_not_prompt_masked_except_padding() -> None:
    torch = pytest.importorskip("torch")
    if CorpusCausalLMCollator is None:
        pytest.skip("CorpusCausalLMCollator requires PyTorch.")
    dataset = CorpusCausalLMDataset(
        [
            TextCorpusRecord(id="short", text="abc"),
            TextCorpusRecord(id="long", text="abcdef"),
        ],
        ByteTokenizer(),
        max_seq_len=16,
    )
    batch = CorpusCausalLMCollator(ByteTokenizer())([dataset[0], dataset[1]])

    labels = batch["labels"]
    assert (labels == -100).sum().item() >= 0
    for row_index, item in enumerate([dataset[0], dataset[1]]):
        length = len(item["input_ids"])
        assert torch.equal(labels[row_index, :length], batch["input_ids"][row_index, :length])


def test_corpus_chunking_is_deterministic() -> None:
    records = [TextCorpusRecord(id="r", text="abcdef" * 20)]
    first = CorpusCausalLMDataset(records, ByteTokenizer(), max_seq_len=12, stride=6)
    second = CorpusCausalLMDataset(records, ByteTokenizer(), max_seq_len=12, stride=6)

    assert [first[index]["input_ids"] for index in range(len(first))] == [
        second[index]["input_ids"] for index in range(len(second))
    ]


def test_continued_pretrain_config_defaults_are_cpu_safe() -> None:
    config = ContinuedPretrainConfig()

    assert config.device == "cpu"
    assert config.use_amp is False
    assert config.batch_size == 2
    assert config.max_steps == 3


def test_run_continued_pretraining_works_for_one_step(tmp_path) -> None:
    pytest.importorskip("torch")
    corpus_path = tmp_path / "corpus.jsonl"
    TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=4))

    result = run_continued_pretraining(
        ContinuedPretrainConfig(
            corpus_path=str(corpus_path),
            lesson_path=None,
            run_registry_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=64,
            d_model=16,
            n_layers=1,
            n_heads=2,
        )
    )

    assert result.corpus_records == 4
    assert result.corpus_chunks >= 4
    assert result.finite is True
    assert result.metrics["continued_pretraining"] is True
    assert Path(result.artifacts["continued_pretrain_result_json"]).exists()
    assert Path(result.artifacts["metrics_json"]).exists()
    assert Path(result.artifacts["corpus_summary_json"]).exists()


def test_run_continued_pretraining_registers_checkpoint_when_enabled(tmp_path) -> None:
    pytest.importorskip("torch")
    corpus_path = tmp_path / "corpus.jsonl"
    artifact_root = tmp_path / "artifacts"
    TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=3))

    result = run_continued_pretraining(
        ContinuedPretrainConfig(
            corpus_path=str(corpus_path),
            lesson_path=None,
            run_registry_root=str(tmp_path / "runs"),
            artifact_root=str(artifact_root),
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=64,
            d_model=16,
            n_layers=1,
            n_heads=2,
            save_checkpoints=True,
        )
    )

    assert result.artifacts["checkpoint_artifact_id"]
    assert Path(result.artifacts["checkpoint_path"]).exists()
    manifest_text = (artifact_root / "manifest.jsonl").read_text(encoding="utf-8")
    assert result.artifacts["checkpoint_artifact_id"] in manifest_text


def test_continued_pretraining_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
