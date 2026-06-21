"""Serious coding bugfix dataset preparation for GPU efficiency runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mopforge.builders import BUG_CATEGORIES, generate_coding_bugfix_lessons
from mopforge.datasets import DatasetRegistry, create_dataset_split, write_split_jsonl
from mopforge.formatting import FIXED_CODE_XML_FORMAT
from mopforge.kts import LessonStore
from mopforge.quality import frame_verified_target_lessons


QUALITY_FORMATS = {"raw", FIXED_CODE_XML_FORMAT}


def prepare_efficiency_dataset(
    *,
    source_path: str | Path = "data/coding_bugfix_efficiency_lessons.jsonl",
    dataset_root: str | Path = "datasets",
    dataset_id: str = "coding_bugfix_efficiency",
    count_per_category: int = 100,
    verify: bool = True,
    timeout_seconds: int = 5,
    split_seed: int = 42,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    overwrite: bool = False,
    quality_format: str = "raw",
) -> dict[str, Any]:
    """Generate, register, split, and materialize a serious lesson dataset."""

    if quality_format not in QUALITY_FORMATS:
        raise ValueError(f"quality_format must be one of: {', '.join(sorted(QUALITY_FORMATS))}.")
    source = Path(source_path)
    if source.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset source already exists: {source}. Pass overwrite=True to replace it."
            )
        source.unlink()
    lessons = generate_coding_bugfix_lessons(
        count_per_category=count_per_category,
        verify=verify,
        timeout_seconds=timeout_seconds,
    )
    if verify:
        failed = [lesson.id for lesson in lessons if not lesson.is_verified]
        if failed:
            raise ValueError(f"Generated dataset has {len(failed)} unverified lessons.")
    if quality_format == FIXED_CODE_XML_FORMAT:
        lessons = frame_verified_target_lessons(
            lessons,
            teacher_source="known_verified_bugfix_target",
            output_format=FIXED_CODE_XML_FORMAT,
            require_verified=verify,
        )
    store = LessonStore(source)
    store.add_many(lessons)

    registry = DatasetRegistry(dataset_root)
    manifest = registry.register_dataset(
        name="Coding Bugfix Efficiency",
        kind="lessons",
        source_paths=[str(source)],
        dataset_id=dataset_id,
        description="Deterministic larger coding bugfix dataset for GPU efficiency comparisons.",
        tags=["coding", "debugging", "gpu-efficiency", f"quality-format:{quality_format}"],
        metadata={
            "count_per_category": count_per_category,
            "bug_categories": list(BUG_CATEGORIES),
            "verified": verify,
            "quality_format": quality_format,
            "split_seed": split_seed,
            "purpose": "warm_sparse_gpu_efficiency",
        },
        copy_files=True,
    )
    split = create_dataset_split(
        manifest,
        train=train_ratio,
        eval=eval_ratio,
        test=test_ratio,
        seed=split_seed,
    )
    version_dir = Path(manifest.metadata["version_dir"])
    materialized_dir = version_dir / "materialized"
    split_paths = {
        split_name: write_split_jsonl(
            manifest,
            split,
            split_name,
            materialized_dir / f"{split_name}.jsonl",
        )
        for split_name in ("train", "eval", "test")
    }
    summary = {
        "dataset_id": manifest.dataset_id,
        "version_id": manifest.version_id,
        "dataset_ref": f"{manifest.dataset_id}@{manifest.version_id}",
        "manifest_path": manifest.metadata["manifest_path"],
        "source_path": str(source),
        "combined_sha256": manifest.combined_sha256,
        "record_count": len(lessons),
        "verified_count": sum(
            1
            for lesson in lessons
            if lesson.verification.get("status") in {"verified", "verified_target"}
        ),
        "count_per_category": count_per_category,
        "quality_format": quality_format,
        "split_id": split.split_id,
        "split_seed": split.seed,
        "split_counts": dict(split.counts),
        "split_paths": split_paths,
    }
    summary_path = version_dir / "efficiency_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
