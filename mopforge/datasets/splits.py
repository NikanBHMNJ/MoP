"""Deterministic dataset splits and materialization helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mopforge.datasets.manifest import DatasetManifest
from mopforge.datasets.registry import DatasetRegistry


@dataclass(slots=True)
class DatasetSplit:
    """A deterministic split for one dataset version."""

    split_id: str
    dataset_id: str
    version_id: str
    seed: int
    ratios: dict[str, float]
    counts: dict[str, int]
    lesson_ids: dict[str, list[str]] = field(default_factory=dict)
    record_indices: dict[str, list[int]] = field(default_factory=dict)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("split_id", "dataset_id", "version_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        if not isinstance(self.ratios, dict) or not isinstance(self.counts, dict):
            raise ValueError("ratios and counts must be dictionaries.")
        if self.created_at == "":
            self.created_at = _now()
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return {
            "split_id": self.split_id,
            "dataset_id": self.dataset_id,
            "version_id": self.version_id,
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "counts": dict(self.counts),
            "lesson_ids": {key: list(value) for key, value in self.lesson_ids.items()},
            "record_indices": {key: list(value) for key, value in self.record_indices.items()},
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSplit":
        """Create a split from a dictionary."""

        return cls(
            split_id=str(data["split_id"]),
            dataset_id=str(data["dataset_id"]),
            version_id=str(data["version_id"]),
            seed=int(data["seed"]),
            ratios={key: float(value) for key, value in data.get("ratios", {}).items()},
            counts={key: int(value) for key, value in data.get("counts", {}).items()},
            lesson_ids={
                key: [str(item) for item in value]
                for key, value in data.get("lesson_ids", {}).items()
            },
            record_indices={
                key: [int(item) for item in value]
                for key, value in data.get("record_indices", {}).items()
            },
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Save split JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSplit":
        """Load split JSON."""

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def create_dataset_split(
    manifest: DatasetManifest,
    train: float = 0.8,
    eval: float = 0.1,
    test: float = 0.1,
    seed: int = 123,
    stratify_by: str | None = None,
) -> DatasetSplit:
    """Create and persist a deterministic split for a dataset manifest."""

    _validate_ratios(train, eval, test)
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    if stratify_by is not None and stratify_by not in {
        "skill",
        "domain",
        "target_module",
        "bug_type",
    }:
        raise ValueError(
            "stratify_by must be skill, domain, target_module, bug_type, or None."
        )
    records = _load_records(manifest)
    indexed = list(enumerate(records))
    rng = random.Random(seed)
    if stratify_by:
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for item in indexed:
            groups.setdefault(_stratify_value(item[1], stratify_by), []).append(item)
        train_items: list[tuple[int, dict[str, Any]]] = []
        eval_items: list[tuple[int, dict[str, Any]]] = []
        test_items: list[tuple[int, dict[str, Any]]] = []
        for group_key in sorted(groups):
            group = groups[group_key]
            rng.shuffle(group)
            child_train, child_eval, child_test = _partition(group, train, eval)
            train_items.extend(child_train)
            eval_items.extend(child_eval)
            test_items.extend(child_test)
        for items in (train_items, eval_items, test_items):
            items.sort(key=lambda item: item[0])
    else:
        shuffled = list(indexed)
        rng.shuffle(shuffled)
        train_items, eval_items, test_items = _partition(shuffled, train, eval)
    split_id = _split_id(seed, train, eval, test, stratify_by)
    buckets = {
        "train": train_items,
        "eval": eval_items,
        "test": test_items,
    }
    lesson_ids: dict[str, list[str]] = {}
    record_indices: dict[str, list[int]] = {}
    if manifest.kind == "lessons" and all(_record_id(record) for _, record in indexed):
        lesson_ids = {
            name: [_record_id(record) for _, record in items if _record_id(record)]
            for name, items in buckets.items()
        }
    else:
        record_indices = {name: [index for index, _ in items] for name, items in buckets.items()}
    split = DatasetSplit(
        split_id=split_id,
        dataset_id=manifest.dataset_id,
        version_id=manifest.version_id,
        seed=seed,
        ratios={"train": float(train), "eval": float(eval), "test": float(test)},
        counts={name: len(items) for name, items in buckets.items()},
        lesson_ids=lesson_ids,
        record_indices=record_indices,
        created_at=_now(),
        metadata={
            "stratify_by": stratify_by,
            "manifest_path": manifest.metadata.get("manifest_path"),
        },
    )
    version_dir = manifest.metadata.get("version_dir")
    if version_dir:
        split.save(Path(version_dir) / "splits" / f"{split.split_id}.json")
    return split


def load_dataset_split(
    dataset_id,
    split_id,
    version_id=None,
    root="datasets",
) -> DatasetSplit:
    """Load a split by dataset/version/split ID."""

    manifest = DatasetRegistry(root).load_manifest(dataset_id, version_id)
    path = Path(manifest.metadata["version_dir"]) / "splits" / f"{split_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split does not exist: {split_id}")
    return DatasetSplit.load(path)


def load_records_for_split(
    manifest: DatasetManifest,
    split: DatasetSplit,
    split_name: str,
) -> list[dict[str, Any]]:
    """Load original JSON records for a split bucket."""

    if split_name not in {"train", "eval", "test"}:
        raise ValueError("split_name must be train, eval, or test.")
    records = _load_records(manifest)
    if split.lesson_ids:
        wanted = set(split.lesson_ids.get(split_name, []))
        return [record for record in records if _record_id(record) in wanted]
    if split.record_indices:
        indices = split.record_indices.get(split_name, [])
        return [records[index] for index in indices if 0 <= index < len(records)]
    raise ValueError("split has no lesson_ids or record_indices.")


def write_split_jsonl(
    manifest: DatasetManifest,
    split: DatasetSplit,
    split_name: str,
    output_path,
) -> str:
    """Materialize a split bucket to JSONL and return the output path."""

    records = load_records_for_split(manifest, split, split_name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    return str(path)


def _validate_ratios(train: float, eval: float, test: float) -> None:
    ratios = [train, eval, test]
    if not all(isinstance(value, (int, float)) and value >= 0 for value in ratios):
        raise ValueError("split ratios must be non-negative numbers.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0.")


def _partition(items, train: float, eval: float):
    total = len(items)
    train_count = int(total * train)
    eval_count = int(total * eval)
    return (
        list(items[:train_count]),
        list(items[train_count : train_count + eval_count]),
        list(items[train_count + eval_count :]),
    )


def _split_id(seed: int, train: float, eval: float, test: float, stratify_by: str | None) -> str:
    base = (
        f"split-seed{seed}-train{round(train * 100):02d}"
        f"-eval{round(eval * 100):02d}-test{round(test * 100):02d}"
    )
    if stratify_by:
        base += f"-by-{stratify_by.replace('_', '-')}"
    return base


def _load_records(manifest: DatasetManifest) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _source_paths_for_manifest(manifest):
        with Path(path).open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if isinstance(record, dict):
                    records.append(record)
    return records


def _source_paths_for_manifest(manifest: DatasetManifest) -> list[str]:
    copied = manifest.metadata.get("copied_source_paths")
    if copied:
        return list(copied)
    return list(manifest.source_paths)


def _record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("lesson_id")
    return str(value) if value is not None else ""


def _stratify_value(record: dict[str, Any], stratify_by: str) -> str:
    if stratify_by in {"skill", "domain"}:
        return str(record.get(stratify_by) or "none")
    if stratify_by == "bug_type":
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and metadata.get("bug_type"):
            return str(metadata["bug_type"])
        return str(record.get("subskill") or "none")
    modules = record.get("target_modules")
    if isinstance(modules, list) and modules:
        return str(modules[0])
    if isinstance(modules, str):
        return modules
    return "none"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
