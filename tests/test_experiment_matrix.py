"""Tests for experiment registry and matrix runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mopforge.cli.main import main as cli_main
from mopforge.configs import (
    MoPForgeConfig,
    dry_run_config,
    experiment_config_from_envelope,
    validate_config_envelope,
)
from mopforge.experiments import (
    ExperimentConfig,
    ExperimentRecord,
    ExperimentRegistry,
    expand_experiment_matrix,
    run_experiment,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson


def make_lesson(lesson_id: str) -> KnowledgeLesson:
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
        metadata={"test_code": "assert add(1, 2) == 3"},
    )


def build_tiny_store(tmp_path) -> None:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))


def sft_envelope(tmp_path, **payload_overrides) -> MoPForgeConfig:
    payload = {
        "mode": "sft_full",
        "lesson_path": str(tmp_path / "lessons.jsonl"),
        "index_path": str(tmp_path / "lessons.sqlite"),
        "run_registry_root": str(tmp_path / "runs"),
        "artifact_root": str(tmp_path / "artifacts"),
        "max_steps": 1,
        "eval_batches": 1,
        "batch_size": 1,
        "max_seq_len": 64,
        "save_checkpoints": False,
        "save_full_checkpoints": False,
    }
    payload.update(payload_overrides)
    return MoPForgeConfig(kind="sft", payload=payload)


def experiment_list_config(tmp_path, *, include_failure: bool = False) -> ExperimentConfig:
    runs = [
        sft_envelope(tmp_path),
        sft_envelope(
            tmp_path,
            mode="sft_adapter",
            model_type="mop_oracle",
            target_modules=["coding"],
            use_fast_adapters=True,
            fast_adapter_names=["coding"],
        ),
    ]
    if include_failure:
        runs.append(sft_envelope(tmp_path, lesson_path=str(tmp_path / "missing.jsonl")))
    return ExperimentConfig(
        name="tiny_experiment_test",
        kind="list",
        runs=[run.to_dict() for run in runs],
        max_runs=len(runs),
        tags=["test"],
    )


def test_experiment_config_validation_and_dict_round_trip(tmp_path) -> None:
    config = experiment_list_config(tmp_path)

    loaded = ExperimentConfig.from_dict(config.to_dict())

    assert loaded.name == config.name
    assert loaded.kind == "list"
    with pytest.raises(ValueError, match="name"):
        ExperimentConfig(name="", kind="list", runs=[sft_envelope(tmp_path).to_dict()])
    with pytest.raises(ValueError, match="kind"):
        ExperimentConfig(name="bad", kind="giant")
    with pytest.raises(ValueError, match="matrix value"):
        ExperimentConfig(
            name="bad",
            base_config=sft_envelope(tmp_path).to_dict(),
            matrix={"payload.mode": "sft_full"},
        )


def test_matrix_expansion_dotted_paths_and_ordering(tmp_path) -> None:
    config = ExperimentConfig(
        name="matrix",
        base_config=sft_envelope(tmp_path).to_dict(),
        matrix={
            "payload.mode": ["sft_full", "sft_adapter"],
            "payload.max_steps": [1, 2],
        },
    )

    expanded = expand_experiment_matrix(config)

    assert [run.payload["mode"] for run in expanded] == [
        "sft_full",
        "sft_full",
        "sft_adapter",
        "sft_adapter",
    ]
    assert [run.payload["max_steps"] for run in expanded] == [1, 2, 1, 2]
    assert expanded[0].metadata["experiment"]["matrix_index"] == 0
    assert expanded[0].metadata["experiment"]["matrix_values"] == {
        "payload.max_steps": 1,
        "payload.mode": "sft_full",
    }


def test_matrix_expansion_max_runs_and_invalid_path(tmp_path) -> None:
    config = ExperimentConfig(
        name="limited",
        base_config=sft_envelope(tmp_path).to_dict(),
        matrix={"payload.mode": ["sft_full", "sft_adapter"], "payload.max_steps": [1, 2]},
        max_runs=3,
    )

    assert len(expand_experiment_matrix(config)) == 3

    bad = ExperimentConfig(
        name="bad-path",
        base_config=sft_envelope(tmp_path).to_dict(),
        matrix={"mode": ["sft_full"]},
    )
    with pytest.raises(ValueError, match="dotted path"):
        expand_experiment_matrix(bad)

    too_large = ExperimentConfig(
        name="too-large",
        base_config=sft_envelope(tmp_path).to_dict(),
        matrix={"payload.max_steps": list(range(129))},
    )
    with pytest.raises(ValueError, match="limit local CPU execution"):
        expand_experiment_matrix(too_large)


def test_explicit_list_experiment_expands(tmp_path) -> None:
    config = experiment_list_config(tmp_path)
    expanded = expand_experiment_matrix(config)

    assert len(expanded) == 2
    assert expanded[1].payload["mode"] == "sft_adapter"
    assert expanded[1].metadata["experiment"]["matrix_index"] == 1


def test_experiment_registry_creates_saves_loads_and_writes_files(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    config = experiment_list_config(tmp_path)
    record = registry.create_experiment(config)
    record.status = "completed"
    record.total_runs = 1
    record.completed_runs = 1
    registry.save_record(record)

    loaded = registry.load_record(record.experiment_id)
    expanded_path = registry.write_expanded_runs(record.experiment_id, expand_experiment_matrix(config))
    summary_path = registry.write_summary(record.experiment_id, {"rows": []})
    csv_path = registry.write_summary_csv(record.experiment_id, [{"experiment_id": record.experiment_id, "index": 0}])
    run_path = registry.write_run_record(record.experiment_id, 0, {"status": "completed"})

    assert loaded.status == "completed"
    assert registry.list_experiments()[0].experiment_id == record.experiment_id
    assert expanded_path.exists()
    assert summary_path.exists()
    assert csv_path.exists()
    assert run_path.exists()


def test_experiment_record_validation() -> None:
    record = ExperimentRecord(
        experiment_id="exp-1",
        name="demo",
        status="created",
        created_at="now",
        updated_at="now",
        total_runs=0,
        completed_runs=0,
        failed_runs=0,
        run_ids=[],
    )

    assert record.to_dict()["experiment_id"] == "exp-1"
    with pytest.raises(ValueError, match="status"):
        ExperimentRecord(
            experiment_id="exp-1",
            name="demo",
            status="weird",
            created_at="now",
            updated_at="now",
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            run_ids=[],
        )


def test_run_experiment_two_sft_runs_and_summary_files(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = run_experiment(
        experiment_list_config(tmp_path),
        registry_root=tmp_path / "experiments",
    )

    assert result.status == "completed"
    assert result.total_runs == 2
    assert result.completed_runs == 2
    assert result.failed_runs == 0
    assert Path(result.summary_path).exists()
    assert Path(result.summary_csv_path).exists()

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary["rows"][0]["experiment_id"] == result.experiment_id
    assert summary["rows"][0]["status"] == "completed"

    with Path(result.summary_csv_path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["result_path"]


def test_run_experiment_catches_failed_child_and_continues(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = run_experiment(
        experiment_list_config(tmp_path, include_failure=True),
        registry_root=tmp_path / "experiments",
    )

    assert result.status == "completed_with_failures"
    assert result.completed_runs == 2
    assert result.failed_runs == 1
    assert result.run_records[-1]["status"] == "failed"
    assert result.run_records[-1]["error"]


def test_experiment_config_mapping_validation_and_dry_run(tmp_path) -> None:
    envelope = MoPForgeConfig(kind="experiment", payload=experiment_list_config(tmp_path).to_dict())

    mapped = experiment_config_from_envelope(envelope)
    messages = validate_config_envelope(envelope)
    summary = dry_run_config(envelope)

    assert mapped.kind == "list"
    assert not [message for message in messages if message.startswith("ERROR:")]
    assert summary["experiment"]["expanded_run_count"] == 2
    assert summary["runnable_locally"] is True


def test_cli_experiment_dry_run_run_list_show_and_default(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    config_path = MoPForgeConfig(
        kind="experiment",
        payload=experiment_list_config(tmp_path).to_dict(),
    ).save(tmp_path / "experiment.json")
    registry_root = tmp_path / "experiments"

    assert cli_main(["experiment", "dry-run", str(config_path)]) == 0
    dry_output = capsys.readouterr().out
    assert '"expanded_run_count": 2' in dry_output

    assert cli_main([
        "experiment",
        "run",
        str(config_path),
        "--registry-root",
        str(registry_root),
    ]) == 0
    run_output = capsys.readouterr().out
    assert "experiment_id=" in run_output
    experiment_id = [
        line.split("=", 1)[1]
        for line in run_output.splitlines()
        if line.startswith("experiment_id=")
    ][0]

    assert cli_main(["experiment", "list", "--registry-root", str(registry_root)]) == 0
    list_output = capsys.readouterr().out
    assert experiment_id in list_output

    assert cli_main([
        "experiment",
        "show",
        experiment_id,
        "--registry-root",
        str(registry_root),
    ]) == 0
    show_output = capsys.readouterr().out
    assert "summary_path=" in show_output

    default_path = tmp_path / "default_experiment.json"
    assert cli_main([
        "config",
        "write-default",
        "experiment_adapter_vs_generated",
        str(default_path),
    ]) == 0
    assert default_path.exists()


def test_experiment_matrix_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
