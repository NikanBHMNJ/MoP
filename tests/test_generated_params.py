"""Tests for tiny generated-parameter / hypernetwork plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.models import (
    ConditionEmbedding,
    GeneratedAdapter,
    GeneratedParameterConfig,
    TinyMoPCausalTransformer,
    condition_names_from_target_modules,
    normalize_condition_names,
)
from mopforge.pretrain import (
    ContinuedPretrainConfig,
    TextCorpusStore,
    build_demo_code_corpus,
    run_continued_pretraining,
)
from mopforge.sft import FinetuneConfig, run_finetune, trainer_config_from_finetune_config
from mopforge.training import (
    DEFAULT_KNOWN_MODULES,
    TinyTrainer,
    TrainerConfig,
    TrainableParameterPolicy,
    apply_trainable_policy,
    build_optimizer_for_trainable_parameters,
    count_parameters,
    infer_parameter_group,
)


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


def build_tiny_store(tmp_path) -> IndexedLessonStore:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))
    return store


def tiny_trainer_config(tmp_path, **overrides) -> TrainerConfig:
    values = dict(
        run_name="generated_params_test",
        model_type="mop_oracle",
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        run_registry_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        use_generated_params=True,
        generated_condition_names=["coding", "debugging", "repair"],
        generated_condition_dim=8,
        generated_rank=2,
        trainable_policy_mode="generated_params_only",
        max_steps=1,
        eval_interval=1,
        checkpoint_interval=1,
        eval_batches=1,
        batch_size=1,
        max_seq_len=96,
        d_model=16,
        n_layers=1,
        n_heads=2,
    )
    values.update(overrides)
    return TrainerConfig(**values)


def tiny_mop(**overrides):
    pytest.importorskip("torch")
    values = dict(
        vocab_size=64,
        d_model=8,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
        module_names=["core", "coding", "debugging"],
    )
    values.update(overrides)
    return TinyMoPCausalTransformer(**values)


def test_generated_parameter_config_validation() -> None:
    config = GeneratedParameterConfig(
        d_model=8,
        condition_dim=4,
        rank=2,
        condition_names=["coding", "debugging"],
    )

    assert config.to_dict()["generator_type"] == "low_rank_adapter"
    with pytest.raises(ValueError, match="d_model"):
        GeneratedParameterConfig(d_model=0)
    with pytest.raises(ValueError, match="condition_dim"):
        GeneratedParameterConfig(d_model=8, condition_dim=0)
    with pytest.raises(ValueError, match="rank"):
        GeneratedParameterConfig(d_model=8, rank=0)
    with pytest.raises(ValueError, match="generator_type"):
        GeneratedParameterConfig(d_model=8, generator_type="giant")
    with pytest.raises(ValueError, match="unique"):
        GeneratedParameterConfig(d_model=8, condition_names=["coding", "coding"])
    with pytest.raises(ValueError, match="residual_scale"):
        GeneratedParameterConfig(d_model=8, residual_scale=float("inf"))


def test_condition_normalization_and_target_module_mapping() -> None:
    names = normalize_condition_names(
        ["coding", "unknown", "coding", ""],
        known_conditions=["coding", "debugging"],
        include_default=True,
    )

    assert names == ["coding"]
    assert condition_names_from_target_modules(
        ["core", "coding", "debugging", "repair", "fast_adapter", "unknown"]
    ) == ["coding", "debugging", "repair", "default"]


def test_condition_embedding_handles_known_unknown_and_none() -> None:
    torch = pytest.importorskip("torch")
    if ConditionEmbedding is None:
        pytest.skip("ConditionEmbedding requires PyTorch.")
    embedding = ConditionEmbedding(["coding", "debugging"], condition_dim=4)

    known = embedding("coding")
    unknown = embedding("unknown")
    none_value = embedding(None)

    assert known.shape == (4,)
    assert unknown is None
    assert none_value is None
    assert isinstance(known, torch.Tensor)


def test_generated_adapter_preserves_shape_and_none_is_unchanged() -> None:
    torch = pytest.importorskip("torch")
    if GeneratedAdapter is None:
        pytest.skip("GeneratedAdapter requires PyTorch.")
    adapter = GeneratedAdapter(
        GeneratedParameterConfig(d_model=8, condition_names=["coding"])
    )
    hidden = torch.randn(2, 3, 8)

    unchanged = adapter(hidden, active_conditions=None)
    changed = adapter(hidden, active_conditions=["coding"])

    assert unchanged.shape == hidden.shape
    assert torch.equal(unchanged, hidden)
    assert changed.shape == hidden.shape
    assert not torch.allclose(changed, hidden)


def test_generated_adapter_single_and_multiple_conditions_are_deterministic() -> None:
    torch = pytest.importorskip("torch")
    if GeneratedAdapter is None:
        pytest.skip("GeneratedAdapter requires PyTorch.")
    torch.manual_seed(123)
    adapter = GeneratedAdapter(
        GeneratedParameterConfig(
            d_model=8,
            condition_names=["coding", "debugging"],
            rank=2,
        )
    )
    adapter.eval()
    hidden = torch.randn(2, 3, 8)

    first_single = adapter(hidden, active_conditions="coding")
    second_single = adapter(hidden, active_conditions="coding")
    first_multi = adapter(hidden, active_conditions=["coding", "debugging"])
    second_multi = adapter(hidden, active_conditions=["coding", "debugging"])

    assert torch.allclose(first_single, second_single)
    assert torch.allclose(first_multi, second_multi)


def test_tiny_mop_forward_generated_disabled_enabled_and_coexisting_adapters() -> None:
    torch = pytest.importorskip("torch")
    if TinyMoPCausalTransformer is None:
        pytest.skip("TinyMoPCausalTransformer requires PyTorch.")
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)

    disabled = tiny_mop(use_generated_params=False)
    disabled_output = disabled(input_ids=input_ids, active_modules=["coding"])
    enabled = tiny_mop(
        use_generated_params=True,
        generated_condition_names=["coding", "debugging"],
        generated_condition_dim=4,
        generated_rank=2,
    )
    enabled_output = enabled(
        input_ids=input_ids,
        active_modules=["coding"],
        active_conditions=["coding"],
    )
    coexisting = tiny_mop(
        use_fast_adapters=True,
        fast_adapter_names=["coding"],
        fast_adapter_bottleneck_dim=4,
        use_generated_params=True,
        generated_condition_names=["coding"],
        generated_condition_dim=4,
        generated_rank=2,
    )
    coexist_output = coexisting(
        input_ids=input_ids,
        active_modules=["coding"],
        active_adapters=["coding"],
        active_conditions=["coding"],
    )

    assert disabled_output["active_conditions"] == [[]]
    assert enabled_output["active_conditions"] == [["coding"]]
    assert coexist_output["active_adapters"] == [["coding"]]
    assert coexist_output["active_conditions"] == [["coding"]]
    assert coexist_output["logits"].shape[:2] == input_ids.shape


def test_generated_parameter_grouping_and_generated_only_policy() -> None:
    model = tiny_mop(
        use_generated_params=True,
        generated_condition_names=["coding", "debugging"],
        generated_condition_dim=4,
        generated_rank=2,
    )

    assert infer_parameter_group(
        "generated_adapter.condition_embedding.embedding.weight"
    ) == "generated_condition_embedding"
    assert infer_parameter_group("generated_adapter.generator.0.weight") == "hypernetwork"

    summaries = apply_trainable_policy(
        model,
        TrainableParameterPolicy(mode="generated_params_only"),
    )
    by_name = {summary.name: summary for summary in summaries}

    assert by_name["generated_condition_embedding"].trainable_params > 0
    assert by_name["hypernetwork"].trainable_params > 0
    assert by_name["shared_core"].trainable_params == 0
    assert by_name["module:coding"].trainable_params == 0
    assert count_parameters(model)["trainable"] < count_parameters(model)["total"]


def test_optimizer_builder_works_with_generated_only_params() -> None:
    torch = pytest.importorskip("torch")
    model = tiny_mop(
        use_generated_params=True,
        generated_condition_names=["coding"],
        generated_condition_dim=4,
        generated_rank=2,
    )
    apply_trainable_policy(model, TrainableParameterPolicy(mode="generated_params_only"))

    optimizer = build_optimizer_for_trainable_parameters(model, learning_rate=1e-3)
    optimizer_params = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]

    assert optimizer_params
    assert all(parameter.requires_grad for parameter in optimizer_params)
    assert sum(parameter.numel() for parameter in optimizer_params) == count_parameters(model)["trainable"]
    assert isinstance(optimizer, torch.optim.AdamW)


def test_tiny_trainer_runs_one_step_with_generated_params_and_checkpoint_metadata(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = TinyTrainer(tiny_trainer_config(tmp_path)).train()
    checkpoint_path = Path(result.artifacts["checkpoint_paths"][-1])
    payload = torch.load(checkpoint_path, map_location="cpu")

    assert result.finite is True
    assert result.metrics["trainable_policy"]["mode"] == "generated_params_only"
    assert result.metrics["generated_metadata"]["enabled"] is True
    assert result.metrics["generated_metadata"]["condition_names"] == [
        "coding",
        "debugging",
        "repair",
    ]
    assert payload["metadata"]["generated_metadata"]["enabled"] is True


def test_sft_generated_mode_maps_and_runs(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    config = FinetuneConfig(
        mode="sft_generated",
        target_modules=["coding"],
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        run_registry_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        max_steps=1,
        eval_batches=1,
        batch_size=1,
        max_seq_len=96,
        save_checkpoints=False,
        generated_rank=2,
    )
    trainer_config = trainer_config_from_finetune_config(config)

    assert trainer_config.trainable_policy_mode == "generated_params_only"
    assert trainer_config.use_generated_params is True
    assert trainer_config.generated_condition_names == ["coding"]

    result = run_finetune(config)

    assert result.mode == "sft_generated"
    assert result.metrics["trainable_policy"]["mode"] == "generated_params_only"
    assert result.metrics["generated_metadata"]["enabled"] is True


def test_continued_pretraining_smoke_with_generated_params(tmp_path) -> None:
    pytest.importorskip("torch")
    corpus_path = tmp_path / "corpus.jsonl"
    TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=3))

    result = run_continued_pretraining(
        ContinuedPretrainConfig(
            corpus_path=str(corpus_path),
            lesson_path=None,
            run_registry_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
            use_generated_params=True,
            generated_condition_names=["coding"],
            generated_condition_dim=8,
            generated_rank=2,
            trainable_policy_mode="generated_params_only",
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=64,
            d_model=16,
            n_layers=1,
            n_heads=2,
            save_checkpoints=False,
        )
    )

    assert result.finite is True
    assert result.metrics["generated_metadata"]["enabled"] is True
    assert result.metrics["trainable_policy"]["mode"] == "generated_params_only"


def test_generated_params_do_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
