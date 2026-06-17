"""Tests for local dataset registry/versioning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.benchmarks import BenchmarkConfig, run_benchmark
from mopforge.cli.main import main as cli_main
from mopforge.configs import (
    MoPForgeConfig,
    dataset_config_from_envelope,
    dry_run_config,
    validate_config_envelope,
)
from mopforge.datasets import (
    DatasetConfig,
    DatasetManifest,
    DatasetRegistry,
    combined_fingerprint,
    compute_dataset_stats,
    create_dataset_split,
    fingerprint_file,
    fingerprint_files,
    load_dataset_split,
    write_split_jsonl,
)
from mopforge.sft import FinetuneConfig, trainer_config_from_finetune_config


def lesson_record(index: int, *, skill: str = "debugging") -> dict:
    return {
        "id": f"lesson-{index}",
        "domain": "coding",
        "skill": skill,
        "subskill": "missing-return",
        "difficulty": 1,
        "target_modules": ["coding", "debugging"],
        "input": "def add(a, b):\n    a + b",
        "expected_output": "def add(a, b):\n    return a + b",
        "verification": {"type": "python_tests", "status": "verified"},
        "metadata": {"source": "unit"},
    }


def write_lessons(path: Path, count: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            skill = "debugging" if index % 2 == 0 else "repair"
            file.write(json.dumps(lesson_record(index, skill=skill), sort_keys=True) + "\n")
    return path


def write_corpus(path: Path, count: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for index in range(count):
            file.write(
                json.dumps(
                    {
                        "id": f"corpus-{index}",
                        "text": f"example text {index}",
                        "source": "unit",
                        "domain": "coding",
                        "language": "python",
                        "metadata": {},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def register_lessons(tmp_path, *, copy_files: bool = False):
    path = write_lessons(tmp_path / "lessons.jsonl")
    registry = DatasetRegistry(tmp_path / "datasets")
    manifest = registry.register_dataset(
        name="coding_bugfix",
        kind="lessons",
        source_paths=[str(path)],
        dataset_id="coding_bugfix",
        copy_files=copy_files,
    )
    return registry, manifest, path


def test_file_fingerprint_deterministic_and_includes_sha_size(tmp_path) -> None:
    path = write_lessons(tmp_path / "lessons.jsonl", count=2)

    first = fingerprint_file(path)
    second = fingerprint_file(path)

    assert first.sha256 == second.sha256
    assert first.size_bytes == path.stat().st_size
    assert len(first.sha256) == 64


def test_combined_fingerprint_deterministic(tmp_path) -> None:
    first = write_lessons(tmp_path / "a.jsonl", count=1)
    second = write_lessons(tmp_path / "b.jsonl", count=2)
    fingerprints = fingerprint_files([str(first), str(second)])

    assert combined_fingerprint(fingerprints) == combined_fingerprint(list(reversed(fingerprints)))


def test_dataset_stats_for_lessons(tmp_path) -> None:
    path = write_lessons(tmp_path / "lessons.jsonl", count=4)

    stats = compute_dataset_stats(path, "lessons")

    assert stats.record_count == 4
    assert stats.domains == {"coding": 4}
    assert stats.skills["debugging"] == 2
    assert stats.target_modules["coding"] == 4
    assert stats.verification_status["verified"] == 4


def test_dataset_stats_for_corpus(tmp_path) -> None:
    path = write_corpus(tmp_path / "corpus.jsonl", count=3)

    stats = compute_dataset_stats(path, "corpus")

    assert stats.record_count == 3
    assert stats.sources == {"unit": 3}
    assert stats.languages == {"python": 3}


def test_dataset_manifest_validation_and_json_round_trip(tmp_path) -> None:
    _, manifest, _ = register_lessons(tmp_path)
    path = tmp_path / "manifest.json"

    manifest.save(path)
    loaded = DatasetManifest.load(path)

    assert loaded.to_dict() == manifest.to_dict()
    with pytest.raises(ValueError, match="dataset_id"):
        DatasetManifest.from_dict({**manifest.to_dict(), "dataset_id": "bad id"})


def test_dataset_registry_register_load_list(tmp_path) -> None:
    registry, manifest, _ = register_lessons(tmp_path)

    record = registry.load_dataset_record("coding_bugfix")

    assert record.latest_version_id == manifest.version_id
    assert registry.load_manifest("coding_bugfix").version_id == manifest.version_id
    assert registry.list_datasets()[0].dataset_id == "coding_bugfix"


def test_snapshot_creates_new_version(tmp_path) -> None:
    registry, manifest, path = register_lessons(tmp_path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(lesson_record(99), sort_keys=True) + "\n")

    snapshot = registry.snapshot_dataset("coding_bugfix")

    assert snapshot.version_id != manifest.version_id
    assert registry.load_dataset_record("coding_bugfix").latest_version_id == snapshot.version_id


def test_dataset_ref_resolution_latest_explicit_and_path(tmp_path) -> None:
    registry, manifest, _ = register_lessons(tmp_path)
    manifest_path = Path(manifest.metadata["manifest_path"])

    assert registry.resolve_dataset_ref("coding_bugfix").version_id == manifest.version_id
    assert registry.resolve_dataset_ref(f"coding_bugfix@{manifest.version_id}").version_id == manifest.version_id
    assert registry.resolve_dataset_ref(str(manifest_path)).dataset_id == "coding_bugfix"


def test_copied_file_mode_copies_source_file(tmp_path) -> None:
    _, manifest, _ = register_lessons(tmp_path, copy_files=True)

    copied = manifest.metadata["copied_source_paths"][0]

    assert Path(copied).exists()
    assert manifest.metadata["file_storage"] == "copied"


def test_deterministic_split_counts_and_same_seed(tmp_path) -> None:
    _, manifest, _ = register_lessons(tmp_path)

    first = create_dataset_split(manifest, train=0.8, eval=0.1, test=0.1, seed=123)
    second = create_dataset_split(manifest, train=0.8, eval=0.1, test=0.1, seed=123)

    assert first.counts == {"train": 8, "eval": 1, "test": 1}
    assert first.lesson_ids == second.lesson_ids
    assert Path(manifest.metadata["version_dir"], "splits", f"{first.split_id}.json").exists()


def test_different_seed_changes_split_order(tmp_path) -> None:
    _, manifest, _ = register_lessons(tmp_path)

    first = create_dataset_split(manifest, seed=1)
    second = create_dataset_split(manifest, seed=2)

    assert first.lesson_ids != second.lesson_ids


def test_ratio_validation_catches_invalid_ratios(tmp_path) -> None:
    _, manifest, _ = register_lessons(tmp_path)

    with pytest.raises(ValueError, match="sum"):
        create_dataset_split(manifest, train=0.7, eval=0.2, test=0.2)


def test_split_materialization_writes_expected_jsonl(tmp_path) -> None:
    _, manifest, _ = register_lessons(tmp_path)
    split = create_dataset_split(manifest, train=0.8, eval=0.1, test=0.1, seed=123)
    output_path = tmp_path / "train.jsonl"

    result_path = write_split_jsonl(manifest, split, "train", output_path)
    rows = [json.loads(line) for line in Path(result_path).read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 8
    assert rows[0]["id"].startswith("lesson-")


def test_dataset_config_envelope_mapping_and_dry_run(tmp_path) -> None:
    path = write_lessons(tmp_path / "lessons.jsonl")
    config = DatasetConfig(
        action="register",
        name="demo",
        kind="lessons",
        source_paths=[str(path)],
        output_root=str(tmp_path / "datasets"),
    )
    envelope = MoPForgeConfig(kind="dataset", payload=config.to_dict())

    mapped = dataset_config_from_envelope(envelope)
    messages = validate_config_envelope(envelope)
    summary = dry_run_config(envelope)

    assert mapped.action == "register"
    assert not [message for message in messages if message.startswith("ERROR:")]
    assert summary["dataset"]["source_count"] == 1


def test_cli_dataset_register_split_list_show_versions_materialize(tmp_path, capsys) -> None:
    path = write_lessons(tmp_path / "lessons.jsonl")
    root = tmp_path / "datasets"

    assert cli_main([
        "dataset",
        "register",
        str(path),
        "--name",
        "coding_bugfix",
        "--kind",
        "lessons",
        "--root",
        str(root),
    ]) == 0
    register_output = capsys.readouterr().out
    assert "dataset_id=coding_bugfix" in register_output

    assert cli_main(["dataset", "list", "--root", str(root)]) == 0
    assert "coding_bugfix" in capsys.readouterr().out

    assert cli_main(["dataset", "show", "coding_bugfix", "--root", str(root)]) == 0
    assert "latest_version_id=" in capsys.readouterr().out

    assert cli_main(["dataset", "versions", "coding_bugfix", "--root", str(root)]) == 0
    assert "sha256=" in capsys.readouterr().out

    assert cli_main([
        "dataset",
        "split",
        "coding_bugfix",
        "--train",
        "0.8",
        "--eval",
        "0.1",
        "--test",
        "0.1",
        "--seed",
        "123",
        "--root",
        str(root),
    ]) == 0
    split_output = capsys.readouterr().out
    split_id = [
        line.split("=", 1)[1]
        for line in split_output.splitlines()
        if line.startswith("split_id=")
    ][0]

    output_path = tmp_path / "train.jsonl"
    assert cli_main([
        "dataset",
        "materialize-split",
        "coding_bugfix",
        "--split-id",
        split_id,
        "--split",
        "train",
        "--output",
        str(output_path),
        "--root",
        str(root),
    ]) == 0
    assert output_path.exists()


def test_config_write_default_dataset_register_lessons(tmp_path, capsys) -> None:
    path = tmp_path / "dataset_register_lessons.json"

    assert cli_main(["config", "write-default", "dataset_register_lessons", str(path)]) == 0
    assert path.exists()
    assert cli_main(["config", "dry-run", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"dataset"' in output


def test_benchmark_config_dataset_ref_metadata_resolves(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    path = write_lessons(tmp_path / "lessons.jsonl")
    DatasetRegistry("datasets").register_dataset(
        name="coding_bugfix",
        kind="lessons",
        source_paths=[str(path)],
        dataset_id="coding_bugfix",
    )
    config = BenchmarkConfig(
        name="dataset_ref_benchmark",
        benchmark_type="parameter_efficiency",
        dataset_ref="coding_bugfix",
        dataset_split="train",
        output_root="benchmarks",
        max_seq_len=64,
    )

    result = run_benchmark(config)

    assert result.status == "completed"
    assert result.metrics["dataset"]["dataset_id"] == "coding_bugfix"
    assert result.metrics["dataset"]["split"] == "train"


def test_sft_dataset_ref_fields_are_mapped_to_trainer_config() -> None:
    config = FinetuneConfig(
        mode="sft_full",
        dataset_ref="coding_bugfix",
        dataset_split="train",
        dataset_version_id="version-a",
    )

    trainer_config = trainer_config_from_finetune_config(config)

    assert trainer_config.dataset_ref == "coding_bugfix"
    assert trainer_config.dataset_split == "train"
    assert trainer_config.dataset_version_id == "version-a"


def test_dataset_layer_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
