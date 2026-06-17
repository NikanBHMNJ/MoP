"""File fingerprint helpers for local dataset versioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class FileFingerprint:
    """Stable fingerprint metadata for one local file."""

    path: str
    size_bytes: int
    sha256: str
    modified_time: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("path must be a non-empty string.")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer.")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hex string.")

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "modified_time": self.modified_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileFingerprint":
        """Create a fingerprint from a dictionary."""

        return cls(
            path=str(data["path"]),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=str(data["sha256"]),
            modified_time=data.get("modified_time"),
        )


def fingerprint_file(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> FileFingerprint:
    """Fingerprint one file without loading it fully into memory."""

    input_path = Path(path).expanduser()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Dataset source file does not exist: {path}")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    digest = hashlib.sha256()
    with input_path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    stat = input_path.stat()
    return FileFingerprint(
        path=str(input_path.resolve()),
        size_bytes=int(stat.st_size),
        sha256=digest.hexdigest(),
        modified_time=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    )


def fingerprint_files(paths: list[str | Path]) -> list[FileFingerprint]:
    """Fingerprint files in the given deterministic order."""

    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list.")
    return [fingerprint_file(path) for path in paths]


def combined_fingerprint(fingerprints: list[FileFingerprint]) -> str:
    """Return a deterministic sha256 over file paths, sizes, and hashes."""

    if not isinstance(fingerprints, list) or not fingerprints:
        raise ValueError("fingerprints must be a non-empty list.")
    digest = hashlib.sha256()
    for fingerprint in sorted(fingerprints, key=lambda item: item.path):
        payload = f"{fingerprint.path}\0{fingerprint.size_bytes}\0{fingerprint.sha256}\n"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()
