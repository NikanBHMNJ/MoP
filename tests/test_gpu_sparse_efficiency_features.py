from __future__ import annotations

from pathlib import Path

import pytest

from mopforge.gpu import (
    GPUTrainer,
    GPUTrainingConfig,
    GPUDataConfig,
    build_cached_activation_dataloaders,
    estimate_active_parameters,
    evaluate_efficiency_gates,
    load_gpu_lesson_splits,
    prepare_efficiency_dataset,
    write_warm_sparse_sweep_configs,
    restore_gpu_checkpoint,
    save_gpu_checkpoint,
    write_activation_cache,
)
from mopforge.cli.main import main as cli_main
from mopforge.configs import MoPForgeConfig
from mopforge.gpu.trainer import apply_activation_checkpointing
from mopforge.models import TinyCausalTransformer, TinyMoPCausalTransformer
from mopforge.training import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    count_parameters,
)
from tests.test_gpu_trainer import _config


torch = pytest.importorskip("torch")


def test_gpu_config_sparse_efficiency_fields_roundtrip() -> None:
    config = GPUTrainingConfig(
        require_device_available=False,
        module_names=["coding", "debugging"],
        always_include_core=False,
        resume_model_only=True,
        save_trainable_only_checkpoints=True,
        activation_cache_path="cache.pt",
        mop_block_type="routed_ffn",
        expert_count=4,
        active_experts=1,
        use_lora_deltas=True,
        lora_rank=4,
        lora_target_modules=["coding"],
        run_generation_eval=True,
        early_stopping_enabled=True,
        early_stopping_patience_evals=3,
        early_stopping_min_delta=0.01,
    )

    loaded = GPUTrainingConfig.from_dict(config.to_dict())

    assert loaded.module_names == ["coding", "debugging"]
    assert loaded.always_include_core is False
    assert loaded.resume_model_only is True
    assert loaded.save_trainable_only_checkpoints is True
    assert loaded.mop_block_type == "routed_ffn"
    assert loaded.lora_rank == 4
    assert loaded.early_stopping_enabled is True
    assert loaded.early_stopping_patience_evals == 3
    assert loaded.early_stopping_min_delta == 0.01


def test_adapters_norm_head_policy_trains_sparse_tail_only() -> None:
    model = _tiny_mop(use_fast_adapters=True)

    summaries = apply_trainable_policy(
        model,
        TrainableParameterPolicy(mode="adapters_norm_head", train_fast_adapters=True),
    )
    by_name = {summary.name: summary for summary in summaries}

    assert by_name["adapter:coding"].trainable_params > 0
    assert by_name["norm"].trainable_params > 0
    assert by_name["lm_head"].trainable_params > 0
    assert by_name["shared_core"].trainable_params == 0
    assert by_name["module:coding"].trainable_params == 0
    assert count_parameters(model)["trainable"] < count_parameters(model)["total"]


def test_frozen_prefix_no_grad_keeps_adapter_gradients() -> None:
    model = _tiny_mop(use_fast_adapters=True)
    apply_trainable_policy(
        model,
        TrainableParameterPolicy(mode="adapters_only", train_fast_adapters=True),
    )
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)
    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        labels=labels,
        active_modules=["coding"],
        active_adapters=["coding"],
    )
    output["loss"].backward()

    assert model.last_forward_metadata["frozen_prefix_no_grad_enabled"] is True
    assert model.last_forward_metadata["frozen_prefix_activation_detached"] is True
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if "shared_blocks" in name
    )
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "fast_adapter_bank.adapters.coding" in name
    )


def test_trainable_only_checkpoint_contains_sparse_delta(tmp_path: Path) -> None:
    model = _tiny_mop(use_fast_adapters=True)
    policy = TrainableParameterPolicy(mode="adapters_norm_head", train_fast_adapters=True)
    apply_trainable_policy(model, policy)
    path = tmp_path / "sparse.pt"

    saved = save_gpu_checkpoint(
        path,
        model=model,
        state={},
        config={},
        trainable_only=True,
        trainable_policy=policy.to_dict(),
    )
    payload = torch.load(saved, map_location="cpu", weights_only=False)

    assert payload["checkpoint_format"] == "mopforge_gpu_train_sparse_v1"
    assert payload["model_state"] is None
    assert payload["trainable_model_state"]
    assert all("shared_blocks" not in key for key in payload["trainable_model_state"])

    restored = _tiny_mop(use_fast_adapters=True)
    apply_trainable_policy(restored, policy)
    metadata = restore_gpu_checkpoint(payload, model=restored, strict_model=False)
    assert "trainable_model_state" in metadata["restored"]


def test_model_only_resume_does_not_restore_trainer_step(tmp_path: Path) -> None:
    first = GPUTrainer(_config(tmp_path, max_steps=1)).train()
    checkpoint = first.artifacts["latest_checkpoint_path"]

    trainer = GPUTrainer(
        _config(
            tmp_path,
            max_steps=2,
            resume_from_checkpoint=checkpoint,
            resume_model_only=True,
        )
    )
    trainer.setup()

    assert trainer.state.global_step == 0
    assert trainer.checkpoint_metadata["resume_model_only"] is True


def test_gpu_trainer_early_stopping_is_opt_in(tmp_path: Path) -> None:
    trainer = GPUTrainer(
        _config(
            tmp_path,
            name="early_stop",
            max_steps=5,
            eval_every_steps=1,
            save_every_steps=5,
            early_stopping_enabled=True,
            early_stopping_patience_evals=1,
            early_stopping_min_delta=0.0,
        )
    )
    trainer.setup()
    trainer.evaluate = lambda: {"eval_loss_mean": 2.0, "eval_batches": 1, "step": trainer.state.global_step}

    result = trainer.train()

    assert result.metrics["global_steps"] == 2
    assert result.metrics["stopped_early"] is True
    assert result.metrics["stop_reason"] == "eval_loss_patience_exhausted"


def test_activation_cache_can_train_sparse_tail(tmp_path: Path) -> None:
    source = GPUTrainer(
        _config(
            tmp_path,
            name="cache_source",
            model_type="mop_oracle",
            use_fast_adapters=True,
            fast_adapter_names=["coding", "debugging", "repair"],
            trainable_policy_mode="adapters_only",
            target_modules=["coding", "debugging", "repair"],
        )
    )
    source.setup()
    cache_path = tmp_path / "activation_cache.pt"
    result = write_activation_cache(
        model=source.model,
        train_loader=source.train_loader,
        eval_loader=source.eval_loader,
        output_path=cache_path,
        runtime=source.runtime,
        max_batches=1,
    )
    assert result["train_records"] > 0

    train_loader, _, metadata = build_cached_activation_dataloaders(
        cache_path,
        micro_batch_size=1,
    )
    batch = next(iter(train_loader))
    assert "hidden_states" in batch
    assert metadata["kind"] == "activation_cache"

    cached = GPUTrainer(
        _config(
            tmp_path,
            name="cache_tail",
            model_type="mop_oracle",
            use_fast_adapters=True,
            fast_adapter_names=["coding", "debugging", "repair"],
            trainable_policy_mode="adapters_norm_head",
            target_modules=["coding", "debugging", "repair"],
            activation_cache_path=str(cache_path),
            max_steps=1,
        )
    ).train()
    assert cached.status == "completed"
    assert cached.metrics["data"]["kind"] == "activation_cache"


def test_activation_cache_rejects_trainable_prefix(tmp_path: Path) -> None:
    source = GPUTrainer(
        _config(
            tmp_path,
            name="unsafe_cache_source",
            model_type="mop_oracle",
            trainable_policy_mode="all",
            target_modules=["coding", "debugging", "repair"],
        )
    )
    source.setup()

    with pytest.raises(ValueError, match="trainable components found"):
        write_activation_cache(
            model=source.model,
            train_loader=source.train_loader,
            eval_loader=source.eval_loader,
            output_path=tmp_path / "unsafe_cache.pt",
            runtime=source.runtime,
            max_batches=1,
        )


def test_routed_ffn_and_lora_delta_forward_smoke() -> None:
    model = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        module_names=["coding", "debugging", "repair"],
        always_include_core=False,
        mop_block_type="routed_ffn",
        expert_count=3,
        active_experts=1,
        use_lora_deltas=True,
        lora_rank=2,
        lora_target_modules=["coding", "debugging", "repair"],
    )
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)

    output = model(input_ids=input_ids, active_modules=["coding"])

    assert output["logits"].shape[:2] == input_ids.shape
    assert output["active_modules"] == [["coding"]]
    assert model.last_forward_metadata["mop_block_type"] == "routed_ffn"


def test_internal_routed_lora_is_zero_init_and_receives_gradients() -> None:
    torch.manual_seed(11)
    base = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        module_names=["coding", "debugging"],
        always_include_core=False,
    )
    lora = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        module_names=["coding", "debugging"],
        always_include_core=False,
        use_lora_deltas=True,
        lora_rank=2,
        lora_target_modules=["coding", "debugging"],
    )
    lora.load_state_dict(base.state_dict(), strict=False)
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    base_logits = base(
        input_ids,
        attention_mask=attention_mask,
        active_modules=["coding"],
    )["logits"]
    initial_logits = lora(
        input_ids,
        attention_mask=attention_mask,
        active_modules=["coding"],
    )["logits"]
    apply_trainable_policy(
        lora,
        TrainableParameterPolicy(
            mode="adapters_norm_head",
            train_lora_deltas=True,
        ),
    )
    output = lora(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        active_modules=["coding"],
    )
    output["loss"].backward()

    assert torch.equal(base_logits, initial_logits)
    assert lora.internal_lora_enabled is True
    assert lora.last_forward_metadata["module_bank_fully_frozen"] is True
    assert lora.last_forward_metadata["frozen_module_bank_no_grad_enabled"] is False
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in lora.named_parameters()
        if "lora_bank" in name and name.endswith("up.weight")
    )
    assert all(
        parameter.grad is None
        for name, parameter in lora.named_parameters()
        if name.endswith("linear1.weight") or name.endswith("linear2.weight")
    )


def test_routed_ffn_has_real_shared_and_expert_parameter_boundaries() -> None:
    model = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=4,
        max_seq_len=16,
        module_names=["coding", "debugging"],
        always_include_core=False,
        mop_block_type="routed_ffn",
        expert_count=2,
        active_experts=1,
        routing_granularity="token",
        shared_depth_ratio=0.5,
    )

    summaries = apply_trainable_policy(
        model,
        TrainableParameterPolicy(mode="core_frozen", train_router=True),
    )
    by_name = {summary.name: summary for summary in summaries}
    active = estimate_active_parameters(model, ["coding"])
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        active_modules=["coding"],
    )
    output["loss"].backward()

    assert model.shared_layer_count == 2
    assert model.routed_layer_count == 2
    assert len(model.shared_blocks.layers) == 2
    assert len(model.routed_blocks) == 2
    assert by_name["shared_core"].trainable_params == 0
    assert by_name["module:coding"].trainable_params > 0
    assert by_name["module:debugging"].trainable_params > 0
    assert by_name["router"].trainable_params > 0
    assert active["active_ratio"] < 1.0
    assert active["routed_module_params"] < active["total_module_params"]
    assert model.last_forward_metadata["frozen_prefix_no_grad_enabled"] is True
    assert model.last_forward_metadata["frozen_routed_blocks_no_grad_enabled"] is False
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "routed_blocks" in name and ".experts.coding." in name
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if "shared_blocks" in name
    )


def test_token_routing_dispatches_only_valid_token_assignments() -> None:
    model = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=16,
        module_names=["coding", "debugging"],
        always_include_core=False,
        mop_block_type="routed_ffn",
        expert_count=2,
        active_experts=1,
        routing_granularity="token",
        shared_depth_ratio=1.0,
    )
    input_ids = torch.tensor([[1, 8, 9, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        active_modules=[[]],
    )

    metadata = model.last_forward_metadata["routed_block_metadata"][0]
    assert metadata["routing_granularity"] == "token"
    assert metadata["valid_token_count"] == 3
    assert metadata["routed_token_assignments"] == 3
    assert sum(metadata["expert_selection_counts"].values()) == 3
    assert metadata["routing_density"] == pytest.approx(0.5)


def test_dense_checkpoint_warm_starts_routed_experts_without_quality_jump() -> None:
    torch.manual_seed(7)
    dense = TinyCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=4,
        max_seq_len=16,
        dropout=0.0,
    )
    routed = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=4,
        max_seq_len=16,
        dropout=0.0,
        module_names=["coding", "debugging"],
        always_include_core=False,
        mop_block_type="routed_ffn",
        expert_count=2,
        active_experts=1,
        routing_granularity="token",
        shared_depth_ratio=0.5,
    )

    metadata = restore_gpu_checkpoint(
        {"model_state": dense.state_dict()},
        model=routed,
        restore_rng=False,
        strict_model=False,
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    dense_logits = dense(input_ids, attention_mask=attention_mask)["logits"]
    routed_logits = routed(
        input_ids,
        attention_mask=attention_mask,
        active_modules=[["coding"], ["debugging"]],
    )["logits"]

    assert metadata["warm_start_adaptation"]["adapted"] is True
    assert metadata["warm_start_adaptation"]["expert_clone_count"] == 4
    assert torch.allclose(dense_logits, routed_logits, atol=1e-6, rtol=1e-6)
    dense_ffn = dense.blocks.layers[2].linear1.weight
    assert torch.equal(routed.routed_blocks[0].experts.coding[0].weight, dense_ffn)
    assert torch.equal(routed.routed_blocks[0].experts.debugging[0].weight, dense_ffn)


def test_native_activation_checkpointing_covers_shared_and_routed_blocks() -> None:
    model = TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=4,
        max_seq_len=16,
        module_names=["coding", "debugging"],
        always_include_core=False,
        mop_block_type="routed_ffn",
        expert_count=2,
        active_experts=1,
        routing_granularity="token",
        shared_depth_ratio=0.5,
    )

    metadata = apply_activation_checkpointing(model, True)
    input_ids = torch.tensor([[1, 8, 9, 2]], dtype=torch.long)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        active_modules=["coding"],
    )
    output["loss"].backward()

    assert metadata["enabled"] is True
    assert metadata["candidate_block_count"] == 4
    assert metadata["applied_block_count"] == 4
    assert not metadata["warnings"]
    assert model.activation_checkpointing_enabled is True
    assert model.token_embedding.weight.grad is not None
    assert model.routed_blocks[0].experts.coding[0].weight.grad is not None


def test_efficiency_gate_report_passes_complete_fake_sparse_run(tmp_path: Path) -> None:
    root = tmp_path / "gpu_runs"
    _write_gate_run(
        root,
        "dense",
        eval_loss=3.1,
        tokens_per_sec=11000.0,
        peak_reserved_gb=1.9,
        gen_pass_rate=1.0,
        trainable_only_checkpoint=False,
    )
    _write_gate_run(
        root,
        "sparse",
        eval_loss=3.2,
        tokens_per_sec=12000.0,
        peak_reserved_gb=0.8,
        gen_pass_rate=0.96,
        trainable_only_checkpoint=True,
    )

    report = evaluate_efficiency_gates(
        dense_run="dense",
        sparse_run="sparse",
        gpu_runs_dir=root,
    )

    assert report["overall_passed"] is True
    assert not report["failed_required_gates"]
    assert not report["unknown_required_gates"]


def test_efficiency_gate_cli_writes_report(tmp_path: Path) -> None:
    root = tmp_path / "gpu_runs"
    _write_gate_run(
        root,
        "dense",
        eval_loss=3.1,
        tokens_per_sec=11000.0,
        peak_reserved_gb=1.9,
        gen_pass_rate=1.0,
        trainable_only_checkpoint=False,
    )
    _write_gate_run(
        root,
        "sparse",
        eval_loss=3.2,
        tokens_per_sec=12000.0,
        peak_reserved_gb=0.8,
        gen_pass_rate=0.96,
        trainable_only_checkpoint=True,
    )
    output = tmp_path / "gate.json"

    assert cli_main(
        [
            "gpu",
            "gate-efficiency",
            "--dense-run",
            "dense",
            "--sparse-run",
            "sparse",
            "--gpu-runs-dir",
            str(root),
            "--output",
            str(output),
        ]
    ) == 0
    assert output.exists()


def test_warm_sparse_sweep_writer_generates_valid_gpu_configs(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"

    paths = write_warm_sparse_sweep_configs(
        output_dir=output_dir,
        base_checkpoint="warm-base-run",
        activation_cache_path="outputs/warm_sparse_cache.pt",
        dataset_ref="coding_bugfix_efficiency@version",
        dataset_split_id="split-seed42-train80-eval10-test10",
        bottlenecks=[64, 128],
        learning_rates=[3e-4],
        max_steps=2000,
    )

    assert len(paths) == 10
    names = {path.name for path in paths}
    assert any("warm_adapters_norm_head_b128" in name for name in names)
    assert any("cached_warm_adapters_norm_head_b64" in name for name in names)
    assert any("warm_lora_norm_head_r16" in name for name in names)
    for path in paths:
        envelope = MoPForgeConfig.load(path)
        assert envelope.kind == "gpu_train"
        gpu_config = GPUTrainingConfig.from_dict(envelope.payload)
        assert gpu_config.resume_model_only is True
        assert gpu_config.save_trainable_only_checkpoints is True
        assert gpu_config.max_steps == 2000
        assert gpu_config.metadata["same_token_budget"] is True
        assert gpu_config.dataset_ref == "coding_bugfix_efficiency@version"
        assert gpu_config.dataset_split_id == "split-seed42-train80-eval10-test10"
    lora_configs = [
        GPUTrainingConfig.from_dict(MoPForgeConfig.load(path).payload)
        for path in paths
        if "warm_lora_norm_head" in path.name
    ]
    assert {config.lora_rank for config in lora_configs} == {4, 8, 16}
    assert all(config.use_lora_deltas for config in lora_configs)
    assert all(not config.use_fast_adapters for config in lora_configs)


def test_warm_sparse_sweep_cli_writes_configs(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep_cli"

    result = cli_main(
        [
            "gpu",
            "write-warm-sparse-sweep",
            "--base-checkpoint",
            "warm-base-run",
            "--output-dir",
            str(output_dir),
            "--bottlenecks",
            "64",
            "--learning-rates",
            "0.001",
        ]
    )

    assert result == 0
    assert len(list(output_dir.glob("*.json"))) == 6


def test_prepare_efficiency_dataset_writes_versioned_fixed_split(tmp_path: Path) -> None:
    result = prepare_efficiency_dataset(
        source_path=tmp_path / "lessons.jsonl",
        dataset_root=tmp_path / "datasets",
        dataset_id="goal47_efficiency",
        count_per_category=2,
        verify=False,
        split_seed=42,
    )

    assert result["record_count"] == 10
    assert result["split_seed"] == 42
    assert sum(result["split_counts"].values()) == 10
    assert Path(result["manifest_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert all(Path(path).exists() for path in result["split_paths"].values())
    train_lessons, eval_lessons, metadata = load_gpu_lesson_splits(
        GPUDataConfig(
            dataset_ref=result["manifest_path"],
            dataset_split_id=result["split_id"],
            max_seq_len=64,
            micro_batch_size=1,
        )
    )
    assert len(train_lessons) == result["split_counts"]["train"]
    assert len(eval_lessons) == result["split_counts"]["eval"]
    assert metadata["fixed_held_out_eval"] is True


def test_prepare_efficiency_dataset_cli(tmp_path: Path) -> None:
    source = tmp_path / "cli_lessons.jsonl"
    root = tmp_path / "datasets"

    result = cli_main(
        [
            "gpu",
            "prepare-efficiency-data",
            "--source-path",
            str(source),
            "--dataset-root",
            str(root),
            "--dataset-id",
            "goal47_cli_efficiency",
            "--count-per-category",
            "1",
            "--no-verify",
        ]
    )

    assert result == 0
    assert source.exists()
    assert (root / "goal47_cli_efficiency" / "dataset.json").exists()


def _tiny_mop(*, use_fast_adapters: bool) -> TinyMoPCausalTransformer:
    if TinyMoPCausalTransformer is None:
        pytest.skip("TinyMoPCausalTransformer requires PyTorch.")
    return TinyMoPCausalTransformer(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=32,
        module_names=["core", "coding", "debugging"],
        use_fast_adapters=use_fast_adapters,
        fast_adapter_names=["coding", "debugging", "repair"],
        fast_adapter_bottleneck_dim=4,
    )


def _write_gate_run(
    root: Path,
    run_id: str,
    *,
    eval_loss: float,
    tokens_per_sec: float,
    peak_reserved_gb: float,
    gen_pass_rate: float,
    trainable_only_checkpoint: bool,
) -> None:
    run_dir = root / run_id
    checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if trainable_only_checkpoint:
        torch.save(
            {
                "checkpoint_format": "mopforge_gpu_train_sparse_v1",
                "model_state": None,
                "trainable_model_state": {"tail.weight": torch.ones(1)},
            },
            checkpoint,
        )
    else:
        torch.save({"checkpoint_format": "mopforge_gpu_train_v1", "model_state": {}}, checkpoint)
    metrics = {
        "status": "completed",
        "latest_train_loss": eval_loss - 0.1,
        "latest_eval_loss": eval_loss,
        "runtime": {"selected_device": "cuda:0", "selected_precision": "bf16"},
        "generation_eval": {
            "enabled": True,
            "gen_pass_rate": gen_pass_rate,
        },
        "model": {
            "routing_mode": "oracle",
        },
        "efficiency": {
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": 10.0,
            "peak_allocated_gb": peak_reserved_gb - 0.1,
            "peak_reserved_gb": peak_reserved_gb,
            "final_reserved_gb": peak_reserved_gb,
            "total_params": 1000,
            "trainable_params": 10 if trainable_only_checkpoint else 1000,
            "trainable_param_ratio": 0.01 if trainable_only_checkpoint else 1.0,
            "active_param_estimate": 500,
            "active_param_ratio": 0.5,
            "active_trainable_param_estimate": 10 if trainable_only_checkpoint else 1000,
            "active_trainable_param_ratio": 0.01 if trainable_only_checkpoint else 1.0,
            "shared_frozen_params": 900 if trainable_only_checkpoint else 0,
            "routed_module_params": 100,
            "estimated_active_flop_ratio": 0.5,
            "checkpoint_size_mb": 0.001,
        },
    }
    result = {
        "run_id": run_id,
        "status": "completed",
        "metrics": metrics,
        "state": {"latest_checkpoint_path": str(checkpoint)},
        "artifacts": {"latest_checkpoint_path": str(checkpoint)},
        "runtime_metadata": {"selected_device": "cuda:0", "selected_precision": "bf16"},
    }
    (run_dir / "metrics.json").write_text(__import__("json").dumps(metrics), encoding="utf-8")
    (run_dir / "gpu_training_result.json").write_text(__import__("json").dumps(result), encoding="utf-8")
