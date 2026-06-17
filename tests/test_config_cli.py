"""Tests for config envelopes and CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.cli.main import main as cli_main
from mopforge.configs import (
    MoPForgeConfig,
    default_pretrain_config,
    default_sft_config,
    default_trainer_config,
    dry_run_config,
    finetune_config_from_envelope,
    get_default_config,
    list_default_config_names,
    load_config_file,
    pretrain_config_from_envelope,
    save_config_file,
    trainer_config_from_envelope,
    validate_config_envelope,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.pretrain import TextCorpusStore, build_demo_code_corpus


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


def test_mopforge_config_dict_round_trip() -> None:
    config = MoPForgeConfig(
        kind="sft",
        payload={"mode": "sft_full", "max_steps": 1},
        metadata={"description": "demo"},
    )

    loaded = MoPForgeConfig.from_dict(config.to_dict())

    assert loaded == config


def test_json_config_save_load(tmp_path) -> None:
    path = tmp_path / "config.json"
    data = {"kind": "sft", "version": "1", "payload": {"mode": "sft_full"}}

    save_config_file(data, path)
    loaded = load_config_file(path)

    assert loaded["kind"] == "sft"
    assert MoPForgeConfig.load(path).kind == "sft"


def test_yaml_config_save_load_or_clear_error(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    data = {"kind": "sft", "version": "1", "payload": {"mode": "sft_full"}}
    try:
        import yaml  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="PyYAML"):
            save_config_file(data, path)
        return

    save_config_file(data, path)
    assert load_config_file(path)["payload"]["mode"] == "sft_full"


def test_default_config_templates_validate() -> None:
    for name in list_default_config_names():
        config = get_default_config(name)
        messages = validate_config_envelope(config)

        assert not [message for message in messages if message.startswith("ERROR:")]


def test_envelope_mapping_to_runtime_configs() -> None:
    sft = default_sft_config("sft_adapter")
    pretrain = default_pretrain_config()
    trainer = default_trainer_config()

    assert finetune_config_from_envelope(sft).mode == "sft_adapter"
    assert pretrain_config_from_envelope(pretrain).model_type == "dense"
    assert trainer_config_from_envelope(trainer).model_type == "mop_oracle"


def test_validation_catches_unknown_kind_and_sft_requirements() -> None:
    unknown = MoPForgeConfig(kind="not-real")
    missing_target = MoPForgeConfig(kind="sft", payload={"mode": "sft_module"})
    missing_generated = MoPForgeConfig(
        kind="sft",
        payload={
            "mode": "sft_generated",
            "target_modules": ["coding"],
            "use_generated_params": False,
        },
    )

    assert any("unknown kind" in message for message in validate_config_envelope(unknown))
    assert any("requires target_modules" in message for message in validate_config_envelope(missing_target))
    assert any("use_generated_params" in message for message in validate_config_envelope(missing_generated))


def test_dry_run_returns_resolved_summary() -> None:
    summary = dry_run_config(default_sft_config("sft_full"))

    assert summary["kind"] == "sft"
    assert summary["runtime_config"]["mode"] == "sft_full"
    assert summary["runnable_locally"] is True
    assert summary["expected_output_roots"]["run_registry_root"] == "runs"


def test_cli_version_and_modes_list(capsys) -> None:
    assert cli_main(["version"]) == 0
    version_output = capsys.readouterr().out
    assert "0.46.0" in version_output

    assert cli_main(["modes", "list"]) == 0
    modes_output = capsys.readouterr().out
    assert "sft_full" in modes_output
    assert "sft_generated" in modes_output


def test_cli_config_write_validate_and_dry_run(tmp_path, capsys) -> None:
    path = tmp_path / "sft_full.json"

    assert cli_main(["config", "write-default", "sft_full", str(path)]) == 0
    assert path.exists()
    assert cli_main(["config", "validate", str(path)]) == 0
    validate_output = capsys.readouterr().out
    assert "validation=valid" in validate_output

    assert cli_main(["config", "dry-run", str(path)]) == 0
    dry_run_output = capsys.readouterr().out
    assert '"runnable_locally": true' in dry_run_output


def test_cli_sft_run_works_for_one_step_tiny_config(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    config = MoPForgeConfig(
        kind="sft",
        payload={
            "mode": "sft_full",
            "lesson_path": str(tmp_path / "lessons.jsonl"),
            "index_path": str(tmp_path / "lessons.sqlite"),
            "run_registry_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts"),
            "max_steps": 1,
            "eval_batches": 1,
            "batch_size": 1,
            "max_seq_len": 96,
            "save_checkpoints": False,
        },
    )
    path = config.save(tmp_path / "sft.json")

    assert cli_main(["sft", "run", str(path)]) == 0
    output = capsys.readouterr().out

    assert "run_id=" in output
    assert "result_path=" in output


def test_cli_pretrain_run_works_for_one_step_tiny_config(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    corpus_path = tmp_path / "corpus.jsonl"
    TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=3))
    config = MoPForgeConfig(
        kind="pretrain",
        payload={
            "corpus_path": str(corpus_path),
            "lesson_path": None,
            "run_registry_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts"),
            "max_steps": 1,
            "eval_batches": 1,
            "batch_size": 1,
            "max_seq_len": 64,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "save_checkpoints": False,
        },
    )
    path = config.save(tmp_path / "pretrain.json")

    assert cli_main(["pretrain", "run", str(path)]) == 0
    output = capsys.readouterr().out

    assert "run_id=" in output
    assert "result_path=" in output


def test_cli_config_layer_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
