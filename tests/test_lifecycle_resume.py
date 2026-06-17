"""Tests for full checkpoint lifecycle and resume plumbing."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mopforge.artifacts import ArtifactManager, CheckpointManager
from mopforge.cli.main import main as cli_main
from mopforge.configs import MoPForgeConfig, dry_run_config, validate_config_envelope
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.lifecycle import (
    CHECKPOINT_FORMAT_VERSION,
    TrainingCheckpointRecord,
    capture_rng_state,
    load_full_training_checkpoint,
    restore_rng_state,
    save_full_training_checkpoint,
)
from mopforge.pretrain import (
    ContinuedPretrainConfig,
    TextCorpusStore,
    build_demo_code_corpus,
    run_continued_pretraining,
)
from mopforge.sft import FinetuneConfig, run_finetune, trainer_config_from_finetune_config
from mopforge.training import TinyTrainer, TrainerConfig


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


def tiny_trainer_config(tmp_path, **overrides) -> TrainerConfig:
    values = {
        "run_name": "resume_trainer_test",
        "model_type": "dense",
        "lesson_path": str(tmp_path / "lessons.jsonl"),
        "index_path": str(tmp_path / "lessons.sqlite"),
        "run_registry_root": str(tmp_path / "runs"),
        "artifact_root": str(tmp_path / "artifacts"),
        "max_steps": 1,
        "eval_interval": 1,
        "eval_batches": 1,
        "batch_size": 1,
        "max_seq_len": 64,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "save_checkpoints": False,
        "save_full_checkpoints": True,
    }
    values.update(overrides)
    return TrainerConfig(**values)


def tiny_finetune_config(tmp_path, **overrides) -> FinetuneConfig:
    values = {
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
        "save_full_checkpoints": True,
    }
    values.update(overrides)
    return FinetuneConfig(**values)


def tiny_pretrain_config(tmp_path, **overrides) -> ContinuedPretrainConfig:
    corpus_path = tmp_path / "corpus.jsonl"
    if not corpus_path.exists():
        TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=3))
    values = {
        "corpus_path": str(corpus_path),
        "lesson_path": None,
        "run_registry_root": str(tmp_path / "runs"),
        "artifact_root": str(tmp_path / "artifacts"),
        "max_steps": 1,
        "eval_batches": 1,
        "batch_size": 1,
        "max_seq_len": 48,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "save_checkpoints": False,
        "save_full_checkpoints": True,
    }
    values.update(overrides)
    return ContinuedPretrainConfig(**values)


def test_rng_capture_restore_keys_and_no_cuda_requirement() -> None:
    state = capture_rng_state()

    assert state["has_python"] is True
    assert "has_numpy" in state
    assert "has_torch" in state
    assert "has_cuda" in state
    random.seed(123)
    before = random.random()
    restore_rng_state(state)
    assert isinstance(before, float)


def test_training_checkpoint_record_validation_and_dict_round_trip(tmp_path) -> None:
    record = TrainingCheckpointRecord(
        checkpoint_id="ckpt-1",
        run_id="run-1",
        step=2,
        training_kind="trainer",
        path=str(tmp_path / "ckpt.pt"),
        metadata={"x": 1},
    )

    loaded = TrainingCheckpointRecord.from_dict(record.to_dict())

    assert loaded == record
    with pytest.raises(ValueError, match="checkpoint_id"):
        TrainingCheckpointRecord(checkpoint_id="", run_id="run", step=0)
    with pytest.raises(ValueError, match="step"):
        TrainingCheckpointRecord(checkpoint_id="ckpt", run_id="run", step=-1)


def test_full_checkpoint_save_load_includes_optimizer_config_tokenizer_and_metadata(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "full.pt"

    record = save_full_training_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        trainer_state={"global_step": 3, "epoch": 0},
        config={"model_type": "dense", "max_steps": 3},
        tokenizer_spec={"tokenizer_type": "byte"},
        parameter_policy={"mode": "all"},
        adapter_metadata={"enabled": False},
        generated_metadata={"enabled": False},
        metadata={"run_id": "run-1", "training_kind": "trainer"},
    )
    payload = load_full_training_checkpoint(path)

    assert record.step == 3
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["model_state_dict"]
    assert payload["optimizer_state_dict"]
    assert payload["trainer_state"]["global_step"] == 3
    assert payload["config"]["max_steps"] == 3
    assert payload["tokenizer_spec"]["tokenizer_type"] == "byte"
    assert payload["rng_state"]["has_cuda"] is False


def test_artifact_manifest_registers_and_finds_latest_full_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    manager = CheckpointManager(ArtifactManager(tmp_path / "artifacts"))

    first = manager.save_full_training_checkpoint(
        model,
        run_id="run-1",
        model_type="dense",
        training_kind="trainer",
        step=1,
        trainer_state={"global_step": 1},
        config={"model_type": "dense"},
        rng_state=capture_rng_state(),
    )
    second = manager.save_full_training_checkpoint(
        model,
        run_id="run-1",
        model_type="dense",
        training_kind="trainer",
        step=2,
        trainer_state={"global_step": 2},
        config={"model_type": "dense"},
        rng_state=capture_rng_state(),
    )

    latest = manager.latest_full_checkpoint(run_id="run-1", training_kind="trainer")

    assert first.metadata["full_checkpoint"] is True
    assert first.metadata["has_rng_state"] is True
    assert latest == second
    assert Path(second.path).exists()


def test_tiny_trainer_saves_full_checkpoint_and_resumes_global_step(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    first = TinyTrainer(tiny_trainer_config(tmp_path, max_steps=1)).train()
    checkpoint_path = first.artifacts["full_checkpoint_paths"][-1]

    resumed = TinyTrainer(
        tiny_trainer_config(
            tmp_path,
            max_steps=2,
            resume_from_checkpoint=checkpoint_path,
        )
    ).train()

    assert first.metrics["full_checkpoint_artifact_ids"]
    assert resumed.final_state["global_step"] == 2
    assert resumed.metrics["resume_global_step"] == 1
    assert resumed.metrics["resume_metadata"]["optimizer_state_restored"] is True
    assert resumed.metrics["resume_metadata"]["rng_state_restored"] is True


def test_full_checkpoint_payload_contains_rng_state_for_tiny_trainer(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(tiny_trainer_config(tmp_path)).train()
    payload = load_full_training_checkpoint(result.artifacts["full_checkpoint_paths"][-1])

    assert payload["rng_state"]["has_python"] is True
    assert payload["optimizer_state_dict"] is not None


def test_finetune_config_maps_resume_fields_into_trainer_config() -> None:
    config = FinetuneConfig(
        mode="sft_full",
        resume_from_checkpoint="checkpoint.pt",
        save_full_checkpoints=True,
        checkpoint_every_steps=2,
    )
    trainer_config = trainer_config_from_finetune_config(config)

    assert trainer_config.resume_from_checkpoint == "checkpoint.pt"
    assert trainer_config.save_full_checkpoints is True
    assert trainer_config.checkpoint_every_steps == 2
    assert trainer_config.training_kind == "sft"


def test_run_finetune_saves_full_checkpoint_and_can_resume(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    first = run_finetune(tiny_finetune_config(tmp_path, max_steps=1))
    checkpoint_path = first.trainer_result["artifacts"]["full_checkpoint_paths"][-1]

    resumed = run_finetune(
        tiny_finetune_config(
            tmp_path,
            max_steps=2,
            resume_from_checkpoint=checkpoint_path,
        )
    )

    assert first.metrics["full_checkpoint_artifact_ids"]
    assert resumed.metrics["global_step"] == 2
    assert resumed.metrics["resume_global_step"] == 1


def test_run_continued_pretraining_saves_full_checkpoint_and_can_resume(tmp_path) -> None:
    pytest.importorskip("torch")
    first = run_continued_pretraining(tiny_pretrain_config(tmp_path, max_steps=1))
    checkpoint_path = first.artifacts["full_checkpoint_path"]

    resumed = run_continued_pretraining(
        tiny_pretrain_config(
            tmp_path,
            max_steps=2,
            resume_from_checkpoint=checkpoint_path,
        )
    )

    assert first.metrics["full_checkpoint_artifact_ids"]
    assert resumed.metrics["global_step"] == 2
    assert resumed.metrics["resume_global_step"] == 1
    assert resumed.metrics["resume_metadata"]["optimizer_state_restored"] is True


def test_cli_train_resume_works_on_tiny_checkpoint(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    first = TinyTrainer(tiny_trainer_config(tmp_path, max_steps=1)).train()
    checkpoint_path = first.artifacts["full_checkpoint_paths"][-1]
    config = MoPForgeConfig(
        kind="trainer",
        payload=tiny_trainer_config(tmp_path, max_steps=2).to_dict(),
    )
    config_path = config.save(tmp_path / "trainer_resume.json")

    assert cli_main(["train", "resume", checkpoint_path, "--config", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "start_step=1" in output
    assert "final_step=2" in output
    assert "result_path=" in output


def test_cli_sft_resume_works_on_tiny_checkpoint(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    first = run_finetune(tiny_finetune_config(tmp_path, max_steps=1))
    checkpoint_path = first.trainer_result["artifacts"]["full_checkpoint_paths"][-1]
    config = MoPForgeConfig(
        kind="sft",
        payload=tiny_finetune_config(tmp_path, max_steps=2).to_dict(),
    )
    config_path = config.save(tmp_path / "sft_resume.json")

    assert cli_main(["sft", "resume", checkpoint_path, "--config", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "start_step=1" in output
    assert "final_step=2" in output


def test_cli_pretrain_resume_works_on_tiny_checkpoint(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    first = run_continued_pretraining(tiny_pretrain_config(tmp_path, max_steps=1))
    checkpoint_path = first.artifacts["full_checkpoint_path"]
    config = MoPForgeConfig(
        kind="pretrain",
        payload=tiny_pretrain_config(tmp_path, max_steps=2).to_dict(),
    )
    config_path = config.save(tmp_path / "pretrain_resume.json")

    assert cli_main(["pretrain", "resume", checkpoint_path, "--config", str(config_path)]) == 0
    output = capsys.readouterr().out

    assert "start_step=1" in output
    assert "final_step=2" in output


def test_config_validation_and_dry_run_include_checkpoint_lifecycle() -> None:
    bad = MoPForgeConfig(
        kind="trainer",
        payload={"checkpoint_every_steps": 0},
    )
    messages = validate_config_envelope(bad)

    assert any("checkpoint_every_steps" in message for message in messages)

    summary = dry_run_config(
        MoPForgeConfig(
            kind="trainer",
            payload={"max_steps": 1, "save_full_checkpoints": True},
        )
    )
    assert summary["checkpointing"]["save_full_checkpoints"] is True


def test_lifecycle_resume_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
