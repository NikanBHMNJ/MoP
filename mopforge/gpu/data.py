"""GPU-aware data loading helpers with CPU-safe defaults."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from mopforge.data import CausalLMCollator, LessonCausalLMDataset
from mopforge.datasets import DatasetRegistry, load_dataset_split, load_records_for_split
from mopforge.datasets.splits import DatasetSplit
from mopforge.kts import KnowledgeLesson, LessonStore
from mopforge.pretrain import CorpusCausalLMCollator, CorpusCausalLMDataset, TextCorpusRecord, TextCorpusStore
from mopforge.runtime import RuntimeContext
from mopforge.tokenization import TokenizerProtocol


@dataclass(slots=True)
class GPUDataConfig:
    dataset_ref: str | None = None
    dataset_split: str | None = None
    dataset_split_id: str | None = None
    lesson_path: str | None = None
    corpus_path: str | None = None
    max_seq_len: int = 1024
    micro_batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    streaming: bool = False
    shuffle_buffer_size: int = 0
    seed: int = 42
    max_examples: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("max_seq_len", "micro_batch_size"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        if type(self.shuffle_buffer_size) is not int or self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be a non-negative integer.")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        if self.max_examples is not None and (type(self.max_examples) is not int or self.max_examples <= 0):
            raise ValueError("max_examples must be a positive integer or None.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUDataConfig":
        return cls(**dict(data))


class StreamingJSONLDataset:
    """Tiny iterable JSONL loader for CPU-safe streaming smoke tests."""

    def __init__(self, path: str | Path, max_examples: int | None = None) -> None:
        self.path = Path(path)
        self.max_examples = max_examples

    def __iter__(self) -> Iterator[dict[str, Any]]:
        count = 0
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if self.max_examples is not None and count >= self.max_examples:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if isinstance(record, dict):
                    count += 1
                    yield record


def build_gpu_dataloaders(
    config: GPUDataConfig,
    tokenizer: TokenizerProtocol,
    runtime: RuntimeContext,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build train/eval DataLoaders for lesson or corpus records."""

    torch = _require_torch()
    pin_memory = bool(config.pin_memory and runtime.device_info.device_type == "cuda")
    prefetch_factor = None if config.num_workers == 0 else 2
    if config.corpus_path:
        records = _load_corpus_records(config)
        train_records, eval_records = _split(records, config.seed)
        train_ds = CorpusCausalLMDataset(train_records, tokenizer, max_seq_len=config.max_seq_len)
        eval_ds = CorpusCausalLMDataset(eval_records, tokenizer, max_seq_len=config.max_seq_len)
        collator = CorpusCausalLMCollator(tokenizer)
        kind = "corpus"
    else:
        train_lessons, eval_lessons, lesson_metadata = load_gpu_lesson_splits(config)
        train_ds = LessonCausalLMDataset(train_lessons, tokenizer, max_length=config.max_seq_len)
        eval_ds = LessonCausalLMDataset(eval_lessons, tokenizer, max_length=config.max_seq_len)
        collator = CausalLMCollator(tokenizer)
        kind = "lessons"
    loader_kwargs = {
        "batch_size": config.micro_batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collator,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = torch.utils.data.DataLoader(train_ds, **loader_kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_ds, **loader_kwargs)
    metadata = {
        "kind": kind,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "record_count": len(records) if config.corpus_path else lesson_metadata["record_count"],
        "max_seq_len": config.max_seq_len,
        "micro_batch_size": config.micro_batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "streaming": config.streaming,
        "dataset_ref": config.dataset_ref,
        "dataset_split": config.dataset_split,
        "dataset_split_id": config.dataset_split_id,
        **({} if config.corpus_path else lesson_metadata),
    }
    return train_loader, eval_loader, metadata


def load_gpu_lesson_splits(
    config: GPUDataConfig,
) -> tuple[list[KnowledgeLesson], list[KnowledgeLesson], dict[str, Any]]:
    if config.dataset_ref and config.dataset_split_id:
        return _load_lessons_from_fixed_split(config)
    lessons, source_metadata = _load_lesson_records(config)
    train_lessons, eval_lessons = _split(lessons, config.seed)
    return train_lessons, eval_lessons, {
        "record_count": len(lessons),
        **source_metadata,
    }


def _load_lesson_records(config: GPUDataConfig) -> tuple[list[KnowledgeLesson], dict[str, Any]]:
    if config.dataset_ref:
        return _load_lessons_from_dataset_ref(config)
    path = Path(config.lesson_path or "data/coding_bugfix_lessons.jsonl")
    lessons = LessonStore(path).load_all()
    return _limit(lessons, config.max_examples, config.seed), {"source_path": str(path)}


def _load_lessons_from_dataset_ref(config: GPUDataConfig) -> tuple[list[KnowledgeLesson], dict[str, Any]]:
    registry = DatasetRegistry()
    manifest = registry.resolve_dataset_ref(config.dataset_ref or "")
    records: list[dict[str, Any]]
    split_name = config.dataset_split
    if split_name:
        bucket = split_name
        split_id = None
        if ":" in split_name:
            split_id, bucket = split_name.split(":", 1)
        elif split_name not in {"train", "eval", "test"}:
            split_id = split_name
            bucket = "train"
        if split_id is None:
            split_files = sorted((Path(manifest.metadata.get("version_dir", "")) / "splits").glob("*.json"))
            if split_files:
                split_id = split_files[0].stem
        if split_id is None:
            raise FileNotFoundError("dataset_split requested but no split JSON was found.")
        split = _load_split_for_manifest(manifest, split_id)
        records = load_records_for_split(manifest, split, bucket)
    else:
        records = []
        for path in manifest.source_paths:
            records.extend(_read_jsonl_dicts(path))
    lessons = [KnowledgeLesson.from_dict(record) for record in records]
    return _limit(lessons, config.max_examples, config.seed), {
        "dataset_id": manifest.dataset_id,
        "version_id": manifest.version_id,
        "dataset_kind": manifest.kind,
    }


def _load_lessons_from_fixed_split(
    config: GPUDataConfig,
) -> tuple[list[KnowledgeLesson], list[KnowledgeLesson], dict[str, Any]]:
    registry = DatasetRegistry()
    manifest = registry.resolve_dataset_ref(config.dataset_ref or "")
    split = _load_split_for_manifest(manifest, config.dataset_split_id or "")
    train_records = load_records_for_split(manifest, split, "train")
    eval_records = load_records_for_split(manifest, split, "eval")
    train_lessons = _limit(
        [KnowledgeLesson.from_dict(record) for record in train_records],
        config.max_examples,
        config.seed,
    )
    eval_lessons = _limit(
        [KnowledgeLesson.from_dict(record) for record in eval_records],
        config.max_examples,
        config.seed,
    )
    if not train_lessons or not eval_lessons:
        raise ValueError("Fixed GPU dataset split must contain non-empty train and eval buckets.")
    return train_lessons, eval_lessons, {
        "record_count": sum(split.counts.values()),
        "dataset_id": manifest.dataset_id,
        "version_id": manifest.version_id,
        "dataset_kind": manifest.kind,
        "dataset_split_id": split.split_id,
        "dataset_split_seed": split.seed,
        "dataset_split_counts": dict(split.counts),
        "fixed_held_out_eval": True,
    }


def _load_split_for_manifest(manifest, split_id: str) -> DatasetSplit:
    version_dir = manifest.metadata.get("version_dir")
    if version_dir:
        path = Path(version_dir) / "splits" / f"{split_id}.json"
        if path.exists():
            return DatasetSplit.load(path)
    return load_dataset_split(
        manifest.dataset_id,
        split_id,
        version_id=manifest.version_id,
    )


def _load_corpus_records(config: GPUDataConfig) -> list[TextCorpusRecord]:
    records = TextCorpusStore(config.corpus_path or "").load_all()
    return _limit(records, config.max_examples, config.seed)


def _read_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                value = json.loads(stripped)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _limit(items: list[Any], max_examples: int | None, seed: int) -> list[Any]:
    values = list(items)
    if max_examples is None or len(values) <= max_examples:
        return values
    rng = random.Random(seed)
    indices = list(range(len(values)))
    rng.shuffle(indices)
    selected = sorted(indices[:max_examples])
    return [values[index] for index in selected]


def _split(items: list[Any], seed: int) -> tuple[list[Any], list[Any]]:
    values = list(items)
    if not values:
        raise ValueError("No records are available for GPU data loading.")
    if len(values) == 1:
        return values, values
    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * 0.8))
    train = shuffled[:split_at]
    eval_items = shuffled[split_at:] or shuffled[:1]
    return train, eval_items


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GPU data loaders.") from exc
    return torch
