"""Dataset statistics for local JSONL datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


KNOWN_DATASET_KINDS = {"lessons", "corpus", "generic_jsonl", "split"}


@dataclass(slots=True)
class DatasetStats:
    """Best-effort JSON-safe stats for a local dataset."""

    record_count: int
    kind: str
    domains: dict[str, int] = field(default_factory=dict)
    skills: dict[str, int] = field(default_factory=dict)
    target_modules: dict[str, int] = field(default_factory=dict)
    verification_status: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.record_count) is not int or self.record_count < 0:
            raise ValueError("record_count must be a non-negative integer.")
        if self.kind not in KNOWN_DATASET_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(sorted(KNOWN_DATASET_KINDS))}.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return {
            "record_count": self.record_count,
            "kind": self.kind,
            "domains": dict(sorted(self.domains.items())),
            "skills": dict(sorted(self.skills.items())),
            "target_modules": dict(sorted(self.target_modules.items())),
            "verification_status": dict(sorted(self.verification_status.items())),
            "sources": dict(sorted(self.sources.items())),
            "languages": dict(sorted(self.languages.items())),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetStats":
        """Create stats from a dictionary."""

        return cls(
            record_count=int(data.get("record_count", 0)),
            kind=str(data["kind"]),
            domains=dict(data.get("domains", {})),
            skills=dict(data.get("skills", {})),
            target_modules=dict(data.get("target_modules", {})),
            verification_status=dict(data.get("verification_status", {})),
            sources=dict(data.get("sources", {})),
            languages=dict(data.get("languages", {})),
            metadata=dict(data.get("metadata", {})),
        )


def compute_dataset_stats(path: str | Path, kind: str) -> DatasetStats:
    """Compute best-effort JSONL stats.

    Malformed non-empty JSONL lines are counted in ``metadata`` instead of
    raising, so record counts and fingerprints can still document local files.
    """

    if kind not in KNOWN_DATASET_KINDS:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    input_path = Path(path)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Dataset source file does not exist: {path}")
    stats = DatasetStats(record_count=0, kind=kind)
    malformed = 0
    empty = 0
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                empty += 1
                continue
            stats.record_count += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            _update_stats(stats, record, kind)
    stats.metadata.update(
        {
            "malformed_lines": malformed,
            "empty_lines": empty,
            "source_path": str(input_path),
        }
    )
    return stats


def _update_stats(stats: DatasetStats, record: dict[str, Any], kind: str) -> None:
    if kind in {"lessons", "generic_jsonl", "split"}:
        _count(stats.domains, record.get("domain"))
        _count(stats.skills, record.get("skill"))
        modules = record.get("target_modules")
        if isinstance(modules, str):
            modules = [modules]
        if isinstance(modules, list):
            for module in modules:
                _count(stats.target_modules, module)
        verification = record.get("verification")
        status = record.get("verification_status")
        if isinstance(verification, dict):
            status = verification.get("status", status)
        _count(stats.verification_status, status)
    if kind in {"corpus", "generic_jsonl", "split"}:
        _count(stats.sources, record.get("source"))
        _count(stats.domains, record.get("domain"))
        _count(stats.languages, record.get("language"))
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        _count(stats.sources, metadata.get("source"))
        _count(stats.languages, metadata.get("language"))


def _count(counter: dict[str, int], value) -> None:
    if value is None:
        return
    label = str(value).strip()
    if not label:
        return
    counter[label] = counter.get(label, 0) + 1
