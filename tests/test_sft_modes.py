"""Tests for FT/SFT training mode API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.sft import (
    FinetuneConfig,
    get_training_mode_spec,
    list_training_modes,
    run_finetune,
    trainer_config_from_finetune_config,
)


def make_lesson(
    lesson_id: str,
    *,
    skill: str = "debugging",
    verification_status: str = "verified",
) -> KnowledgeLesson:
    return KnowledgeLesson(
        id=lesson_id,
        domain="coding",
        skill=skill,
        subskill="missing-return",
        difficulty=1,
        target_modules=["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "python_tests", "status": verification_status},
        metadata={"test_code": "assert add(1, 2) == 3"},
    )


def build_tiny_store(tmp_path, *, include_repair: bool = False) -> IndexedLessonStore:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))
    if include_repair:
        store.add(
            make_lesson(
                "repair-a",
                skill="repair",
                verification_status="verified_target",
            )
        )
    return store


def ft_config(tmp_path, **overrides) -> FinetuneConfig:
    values = dict(
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        run_registry_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        max_steps=1,
        eval_batches=1,
        batch_size=1,
        max_seq_len=128,
        save_checkpoints=False,
    )
    values.update(overrides)
    return FinetuneConfig(**values)


def test_training_modes_are_listed() -> None:
    modes = list_training_modes()

    assert modes == [
        "sft_full",
        "sft_module",
        "sft_adapter",
        "sft_generated",
        "sft_router",
        "repair_sft",
        "continued_pretraining_smoke",
    ]


def test_mode_specs_return_expected_policy_modes() -> None:
    assert get_training_mode_spec("sft_full").expected_policy_mode == "all"
    assert get_training_mode_spec("sft_module").expected_policy_mode == "target_modules_only"
    assert get_training_mode_spec("sft_adapter").expected_policy_mode == "fast_adapters_only"
    assert get_training_mode_spec("sft_generated").expected_policy_mode == "generated_params_only"
    assert get_training_mode_spec("sft_router").expected_policy_mode == "router_only"
    assert get_training_mode_spec("repair_sft").expected_policy_mode == "all"


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported training mode"):
        FinetuneConfig(mode="rlhf")


def test_sft_module_requires_targets() -> None:
    with pytest.raises(ValueError, match="requires target_modules"):
        FinetuneConfig(mode="sft_module")


def test_sft_adapter_enables_and_maps_adapters() -> None:
    config = FinetuneConfig(mode="sft_adapter", target_modules=["coding", "debugging"])

    assert config.use_fast_adapters is True
    assert config.fast_adapter_names == ["coding", "debugging"]

    static_config = FinetuneConfig(
        mode="sft_adapter",
        fast_adapter_names=["coding"],
    )
    trainer_config = trainer_config_from_finetune_config(static_config)

    assert trainer_config.adapter_from_target_modules is False
    assert trainer_config.active_adapters == ["coding"]


def test_sft_generated_enables_and_maps_conditions() -> None:
    config = FinetuneConfig(mode="sft_generated", target_modules=["coding", "debugging"])

    assert config.use_generated_params is True
    assert config.generated_condition_names == ["coding", "debugging"]

    trainer_config = trainer_config_from_finetune_config(config)

    assert trainer_config.model_type == "mop_oracle"
    assert trainer_config.use_generated_params is True
    assert trainer_config.generated_condition_names == ["coding", "debugging"]
    assert trainer_config.trainable_policy_mode == "generated_params_only"


def test_trainer_config_mapping_sft_full() -> None:
    config = trainer_config_from_finetune_config(FinetuneConfig(mode="sft_full"))

    assert config.trainable_policy_mode == "all"
    assert config.model_type == "dense"


def test_trainer_config_mapping_sft_module() -> None:
    config = trainer_config_from_finetune_config(
        FinetuneConfig(mode="sft_module", target_modules=["coding"])
    )

    assert config.model_type == "mop_oracle"
    assert config.trainable_policy_mode == "target_modules_only"
    assert config.trainable_target_modules == ["coding"]
    assert config.target_modules == ["coding"]


def test_trainer_config_mapping_sft_adapter() -> None:
    config = trainer_config_from_finetune_config(
        FinetuneConfig(mode="sft_adapter", target_modules=["coding"])
    )

    assert config.model_type == "mop_oracle"
    assert config.use_fast_adapters is True
    assert config.fast_adapter_names == ["coding"]
    assert config.trainable_policy_mode == "fast_adapters_only"


def test_trainer_config_mapping_repair_sft() -> None:
    config = trainer_config_from_finetune_config(FinetuneConfig(mode="repair_sft"))

    assert config.curriculum_strategy == "repair_boosted"
    assert config.curriculum_skills == ["repair"]
    assert config.trainable_policy_mode == "all"


def test_run_finetune_works_for_one_step_sft_full(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = run_finetune(ft_config(tmp_path, mode="sft_full"))

    assert result.mode == "sft_full"
    assert result.metrics["finetune_mode"] == "sft_full"
    assert result.metrics["trainable_policy"]["mode"] == "all"
    assert Path(result.artifacts["finetune_result_json"]).exists()


def test_run_finetune_works_for_one_step_sft_module(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = run_finetune(
        ft_config(tmp_path, mode="sft_module", target_modules=["coding"])
    )

    assert result.mode == "sft_module"
    assert result.metrics["trainable_policy"]["mode"] == "target_modules_only"
    assert result.metrics["parameter_counts"]["trainable"] > 0
    assert result.metrics["parameter_counts"]["frozen"] > 0


def test_run_finetune_works_for_one_step_sft_adapter(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = run_finetune(
        ft_config(
            tmp_path,
            mode="sft_adapter",
            target_modules=["coding"],
            fast_adapter_names=["coding", "debugging"],
        )
    )

    assert result.mode == "sft_adapter"
    assert result.metrics["trainable_policy"]["mode"] == "fast_adapters_only"
    assert result.metrics["adapter_metadata"]["enabled"] is True
    assert result.metrics["parameter_counts"]["trainable"] > 0


def test_finetune_result_json_is_written_and_contains_mode_metadata(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = run_finetune(ft_config(tmp_path, mode="sft_full"))
    loaded = json.loads(Path(result.artifacts["finetune_result_json"]).read_text(encoding="utf-8"))

    assert loaded["mode"] == "sft_full"
    assert loaded["mode_spec"]["objective"] == "supervised input -> expected_output"
    assert loaded["metrics"]["finetune_expected_policy_mode"] == "all"


def test_sft_api_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
