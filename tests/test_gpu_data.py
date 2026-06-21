import json
from pathlib import Path

import pytest

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.gpu import GPUDataConfig, StreamingJSONLDataset, build_gpu_dataloaders
from mopforge.kts import LessonStore
from mopforge.runtime import RuntimeConfig, build_runtime_context
from mopforge.tokenization import TokenizerSpec, build_tokenizer


pytest.importorskip("torch")


def test_gpu_data_loader_pin_memory_safe_on_cpu_and_max_examples(tmp_path):
    path = tmp_path / "lessons.jsonl"
    LessonStore(path).add_many(
        lesson for lesson in generate_coding_bugfix_lessons(count_per_category=1) if lesson.is_verified
    )
    runtime = build_runtime_context(RuntimeConfig(device="cpu"))
    train, eval_loader, meta = build_gpu_dataloaders(
        GPUDataConfig(lesson_path=str(path), max_seq_len=64, micro_batch_size=1, max_examples=3, pin_memory=True),
        build_tokenizer(TokenizerSpec()),
        runtime,
    )
    assert meta["pin_memory"] is False
    assert meta["record_count"] == 3
    assert next(iter(train))["input_ids"].shape[0] == 1
    assert next(iter(eval_loader))["input_ids"].shape[0] == 1
    assert meta["sequence_length_statistics"]["train"]["examples"] == len(train.dataset)
    assert meta["sequence_length_statistics"]["train"]["max_seq_len"] == 64
    assert meta["sequence_length_statistics"]["train"]["truncated_examples"] > 0


def test_streaming_jsonl_dataset_reads_deterministically(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text("\n".join(json.dumps({"id": i}) for i in range(3)), encoding="utf-8")
    assert [item["id"] for item in StreamingJSONLDataset(path, max_examples=2)] == [0, 1]


def test_train_loader_reshuffles_each_epoch_reproducibly(tmp_path) -> None:
    path = tmp_path / "shuffle_lessons.jsonl"
    LessonStore(path).add_many(
        generate_coding_bugfix_lessons(count_per_category=4, verify=False)
    )
    runtime = build_runtime_context(RuntimeConfig(device="cpu"))
    config = GPUDataConfig(
        lesson_path=str(path),
        max_seq_len=64,
        micro_batch_size=1,
        shuffle_train=True,
        shuffle_seed=17,
    )
    tokenizer = build_tokenizer(TokenizerSpec())
    first_loader, _, metadata = build_gpu_dataloaders(config, tokenizer, runtime)
    second_loader, _, _ = build_gpu_dataloaders(config, tokenizer, runtime)

    first_epoch = [batch["lesson_id"][0] for batch in first_loader]
    second_epoch = [batch["lesson_id"][0] for batch in first_loader]
    repeated_first = [batch["lesson_id"][0] for batch in second_loader]
    repeated_second = [batch["lesson_id"][0] for batch in second_loader]

    assert first_epoch == repeated_first
    assert second_epoch == repeated_second
    assert first_epoch != second_epoch
    assert metadata["shuffle_train"] is True
    assert metadata["shuffle_seed"] == 17
