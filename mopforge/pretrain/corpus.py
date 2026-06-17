"""JSONL text corpus records for continued-pretraining smoke runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mopforge.kts import KnowledgeLesson


@dataclass(slots=True)
class TextCorpusRecord:
    """One raw or semi-structured text record for causal-LM pretraining."""

    id: str
    text: str
    source: str = "manual"
    domain: str | None = None
    language: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill defaults and validate this record."""

        if self.created_at is None:
            self.created_at = _now()
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if this record is malformed."""

        _require_non_empty(self.id, "id")
        _require_non_empty(self.text, "text")
        _require_non_empty(self.source, "source")
        if self.domain is not None and not isinstance(self.domain, str):
            raise ValueError("domain must be a string or None.")
        if self.language is not None and not isinstance(self.language, str):
            raise ValueError("language must be a string or None.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        try:
            json.dumps(self.metadata, sort_keys=True)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-compatible.") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable corpus record."""

        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "domain": self.domain,
            "language": self.language,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextCorpusRecord":
        """Create a corpus record from a dictionary."""

        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data.get("source", "manual")),
            domain=data.get("domain"),
            language=data.get("language"),
            created_at=data.get("created_at"),
            metadata=dict(data.get("metadata", {})),
        )


class TextCorpusStore:
    """JSONL-backed text corpus store."""

    def __init__(self, path: str | Path) -> None:
        """Create a store at ``path``."""

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: TextCorpusRecord) -> None:
        """Append one record, rejecting duplicate IDs."""

        record.validate()
        if self.get(record.id) is not None:
            raise ValueError(f"Duplicate corpus record id: {record.id}")
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def add_many(self, records: list[TextCorpusRecord]) -> int:
        """Append many records and return the number added."""

        records = list(records)
        existing_ids = {record.id for record in self.load_all()}
        new_ids = set()
        for record in records:
            record.validate()
            if record.id in existing_ids or record.id in new_ids:
                raise ValueError(f"Duplicate corpus record id: {record.id}")
            new_ids.add(record.id)
        with self.path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return len(records)

    def load_all(self) -> list[TextCorpusRecord]:
        """Load all corpus records in JSONL order."""

        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(TextCorpusRecord.from_dict(json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.path} at line {line_number}: {exc.msg}."
                    ) from exc
        return records

    def get(self, record_id: str) -> TextCorpusRecord | None:
        """Return one record by ID."""

        for record in self.load_all():
            if record.id == record_id:
                return record
        return None

    def count(self) -> int:
        """Return record count."""

        return len(self.load_all())

    def filter(
        self,
        *,
        domain: str | None = None,
        language: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextCorpusRecord]:
        """Return records matching simple deterministic filters."""

        records = self.load_all()
        filtered = []
        for record in records:
            if domain is not None and record.domain != domain:
                continue
            if language is not None and record.language != language:
                continue
            if source is not None and record.source != source:
                continue
            if metadata is not None:
                if any(record.metadata.get(key) != value for key, value in metadata.items()):
                    continue
            filtered.append(record)
        return filtered

    def export_json(self, path: str | Path) -> Path:
        """Export all records to a JSON array."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                [record.to_dict() for record in self.load_all()],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output_path


def build_corpus_from_lessons(
    lessons: list[KnowledgeLesson],
    *,
    include_input: bool = True,
    include_expected_output: bool = True,
    include_metadata: bool = False,
) -> list[TextCorpusRecord]:
    """Convert structured lessons into raw text/code corpus records."""

    records = []
    for lesson in lessons:
        lesson.validate()
        pieces = [
            f"Domain: {lesson.domain}",
            f"Skill: {lesson.skill}",
            f"Subskill: {lesson.subskill or ''}",
        ]
        if include_input:
            pieces.append("Input:\n" + lesson.input)
        if include_expected_output:
            pieces.append("Expected output:\n" + lesson.expected_output)
        if include_metadata:
            pieces.append(
                "Metadata:\n"
                + json.dumps(lesson.metadata, indent=2, sort_keys=True)
            )
        text = "\n\n".join(piece for piece in pieces if piece.strip())
        records.append(
            TextCorpusRecord(
                id=f"lesson-corpus-{lesson.id}",
                text=text,
                source="lesson",
                domain=lesson.domain,
                language=lesson.metadata.get("language"),
                metadata={
                    "lesson_id": lesson.id,
                    "skill": lesson.skill,
                    "verification_status": lesson.verification.get("status"),
                    "target_modules": list(lesson.target_modules),
                },
            )
        )
    return records


def build_demo_code_corpus(count: int = 20) -> list[TextCorpusRecord]:
    """Build a tiny deterministic Python/code explanation corpus."""

    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer.")
    records = []
    templates = [
        (
            "missing-return",
            "def add(a, b):\n    return a + b",
            "A function should return the computed value explicitly.",
        ),
        (
            "bounds-check",
            "def first(items):\n    return items[0] if items else None",
            "Check whether a sequence is empty before reading index zero.",
        ),
        (
            "accumulator",
            "def total(values):\n    acc = 0\n    for value in values:\n        acc += value\n    return acc",
            "Initialize accumulators with the identity value for the operation.",
        ),
        (
            "base-case",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "Recursive functions need a base case that stops recursion.",
        ),
    ]
    for index in range(count):
        topic, code, note = templates[index % len(templates)]
        records.append(
            TextCorpusRecord(
                id=f"demo-code-corpus-{index:03d}",
                text=f"# Topic: {topic}\n{code}\n\nExplanation: {note}",
                source="demo",
                domain="coding",
                language="python",
                metadata={"topic": topic, "index": index},
            )
        )
    return records


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
