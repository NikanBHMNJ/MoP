"""Local dataset registry and versioning helpers."""

from mopforge.datasets.fingerprint import (
    FileFingerprint,
    combined_fingerprint,
    fingerprint_file,
    fingerprint_files,
)
from mopforge.datasets.manifest import (
    DATASET_ACTIONS,
    DatasetConfig,
    DatasetManifest,
    slugify_dataset_id,
)
from mopforge.datasets.registry import DatasetRecord, DatasetRegistry
from mopforge.datasets.splits import (
    DatasetSplit,
    create_dataset_split,
    load_dataset_split,
    load_records_for_split,
    write_split_jsonl,
)
from mopforge.datasets.stats import (
    KNOWN_DATASET_KINDS,
    DatasetStats,
    compute_dataset_stats,
)

__all__ = [
    "DATASET_ACTIONS",
    "KNOWN_DATASET_KINDS",
    "DatasetConfig",
    "DatasetManifest",
    "DatasetRecord",
    "DatasetRegistry",
    "DatasetSplit",
    "DatasetStats",
    "FileFingerprint",
    "combined_fingerprint",
    "compute_dataset_stats",
    "create_dataset_split",
    "fingerprint_file",
    "fingerprint_files",
    "load_dataset_split",
    "load_records_for_split",
    "slugify_dataset_id",
    "write_split_jsonl",
]
