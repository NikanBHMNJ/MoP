"""Tests for Goal 21 tokenizer abstraction plumbing."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from mopforge.data import (
    CausalLMCollator,
    LessonCausalLMDataset,
    RouterCollator,
    RouterDataset,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.pretrain import (
    ContinuedPretrainConfig,
    CorpusCausalLMCollator,
    CorpusCausalLMDataset,
    TextCorpusRecord,
    TextCorpusStore,
    build_demo_code_corpus,
    run_continued_pretraining,
)
from mopforge.sft import FinetuneConfig, run_finetune
from mopforge.tokenization import (
    ByteTokenizer,
    HFTokenizerWrapper,
    TokenizerSpec,
    build_tokenizer,
)
from mopforge.tokenization.byte_tokenizer import ByteTokenizer as OldByteTokenizer
from mopforge.training import DEFAULT_KNOWN_MODULES, TrainerConfig


class TinyGenericTokenizer:
    """Small tokenizer with custom pad and no BOS/EOS for generic tests."""

    pad_token_id = 99
    bos_token_id = None
    eos_token_id = None
    unk_token_id = None
    vocab_size = 128

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [3 + (ord(ch) % 90) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr((int(token_id) - 3) % 90) for token_id in token_ids)


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


def build_store(tmp_path) -> None:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))


def test_tokenizer_spec_validation_and_json_round_trip(tmp_path) -> None:
    spec = TokenizerSpec(
        tokenizer_type="byte",
        vocab_size=259,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        metadata={"source": "test"},
    )

    path = spec.save_json(tmp_path / "tokenizer_spec.json")
    loaded = TokenizerSpec.load_json(path)

    assert loaded == spec
    assert TokenizerSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="tokenizer_type"):
        TokenizerSpec(tokenizer_type="")
    with pytest.raises(ValueError, match="name_or_path"):
        TokenizerSpec(tokenizer_type="hf")
    with pytest.raises(ValueError, match="metadata"):
        TokenizerSpec(metadata={"bad": object()})


def test_byte_tokenizer_buildable_from_spec_and_old_import_path() -> None:
    tokenizer = build_tokenizer(TokenizerSpec(tokenizer_type="byte"))

    assert isinstance(tokenizer, ByteTokenizer)
    assert OldByteTokenizer is ByteTokenizer
    assert tokenizer.to_spec().pad_token_id == 0


def test_unicode_encode_decode_round_trip_still_works() -> None:
    tokenizer = ByteTokenizer()
    text = "print('hi')\nUnicode: سلام"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_generic_tokenizer_works_with_lesson_dataset_without_bos_eos() -> None:
    tokenizer = TinyGenericTokenizer()
    dataset = LessonCausalLMDataset([make_lesson()], tokenizer, max_length=64)

    item = dataset[0]

    assert item["input_ids"]
    assert item["input_ids"][0] != tokenizer.bos_token_id
    assert len(item["input_ids"]) == len(item["labels"])


def test_generic_tokenizer_works_with_corpus_dataset() -> None:
    tokenizer = TinyGenericTokenizer()
    dataset = CorpusCausalLMDataset(
        [TextCorpusRecord(id="r", text="abc def", source="test")],
        tokenizer,
        max_seq_len=8,
    )

    assert len(dataset) >= 1
    assert dataset[0]["labels"] == dataset[0]["input_ids"]


def test_generic_tokenizer_works_with_router_dataset() -> None:
    tokenizer = TinyGenericTokenizer()
    dataset = RouterDataset(
        [make_lesson()],
        tokenizer,
        known_modules=DEFAULT_KNOWN_MODULES,
        max_length=64,
    )

    item = dataset[0]

    assert item["input_ids"]
    assert item["module_mask"] == [1, 1, 1, 0, 0, 0, 0]


def test_collators_use_tokenizer_pad_token() -> None:
    if CausalLMCollator is None or CorpusCausalLMCollator is None or RouterCollator is None:
        pytest.skip("PyTorch collators are unavailable.")

    tokenizer = TinyGenericTokenizer()
    lesson_dataset = LessonCausalLMDataset(
        [make_lesson("a"), make_lesson("b")],
        tokenizer,
        max_length=32,
    )
    causal_first = dict(lesson_dataset[0])
    causal_first["input_ids"] = causal_first["input_ids"][:4]
    causal_first["labels"] = causal_first["labels"][:4]
    causal_first["attention_mask"] = causal_first["attention_mask"][:4]
    causal_batch = CausalLMCollator(tokenizer)([causal_first, lesson_dataset[1]])

    corpus_dataset = CorpusCausalLMDataset(
        [
            TextCorpusRecord(id="short", text="a", source="test"),
            TextCorpusRecord(id="long", text="abcdef", source="test"),
        ],
        tokenizer,
        max_seq_len=16,
    )
    corpus_first = dict(corpus_dataset[0])
    corpus_first["input_ids"] = corpus_first["input_ids"][:1]
    corpus_first["labels"] = corpus_first["labels"][:1]
    corpus_first["attention_mask"] = corpus_first["attention_mask"][:1]
    corpus_batch = CorpusCausalLMCollator(tokenizer)([corpus_first, corpus_dataset[1]])

    router_dataset = RouterDataset(
        [make_lesson("a"), make_lesson("b")],
        tokenizer,
        known_modules=DEFAULT_KNOWN_MODULES,
        max_length=32,
    )
    router_first = dict(router_dataset[0])
    router_first["input_ids"] = router_first["input_ids"][:3]
    router_first["attention_mask"] = router_first["attention_mask"][:3]
    router_batch = RouterCollator(tokenizer)([router_first, router_dataset[1]])

    assert (causal_batch["input_ids"] == tokenizer.pad_token_id).any().item()
    assert (corpus_batch["input_ids"] == tokenizer.pad_token_id).any().item()
    assert (router_batch["input_ids"] == tokenizer.pad_token_id).any().item()


def test_trainer_config_default_tokenizer_fields_are_cpu_safe() -> None:
    config = TrainerConfig()

    assert config.tokenizer_type == "byte"
    assert config.tokenizer_name_or_path is None
    assert config.tokenizer_spec_path is None
    assert config.device == "cpu"


def test_run_finetune_writes_tokenizer_spec_json(tmp_path) -> None:
    pytest.importorskip("torch")
    build_store(tmp_path)

    result = run_finetune(
        FinetuneConfig(
            lesson_path=str(tmp_path / "lessons.jsonl"),
            index_path=str(tmp_path / "lessons.sqlite"),
            run_registry_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=96,
            save_checkpoints=False,
        )
    )

    spec_path = Path(result.trainer_result["artifacts"]["tokenizer_spec_json"])
    spec_data = json.loads(spec_path.read_text(encoding="utf-8"))

    assert spec_path.exists()
    assert spec_data["tokenizer_type"] == "byte"
    assert result.metrics["tokenizer_spec"]["vocab_size"] == 259


def test_run_continued_pretraining_writes_tokenizer_spec_json(tmp_path) -> None:
    pytest.importorskip("torch")
    corpus_path = tmp_path / "corpus.jsonl"
    TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=3))

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
            save_checkpoints=False,
        )
    )

    assert Path(result.artifacts["tokenizer_spec_json"]).exists()
    assert result.metrics["tokenizer_spec"]["tokenizer_type"] == "byte"


def test_hf_wrapper_missing_dependency_error_is_clear(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in {"transformers", "tokenizers"}:
            raise ImportError("blocked optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install transformers"):
        HFTokenizerWrapper("local-tokenizer-only")


def test_tokenizer_abstraction_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
