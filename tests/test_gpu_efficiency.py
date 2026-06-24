from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.cli.main import main as cli_main
from mopforge.configs import MoPForgeConfig, gpu_training_config_from_envelope
from mopforge.gpu import GPUTrainer
from mopforge.gpu.memory import cuda_memory_metrics, reset_cuda_peak_memory
from mopforge.gpu.validation import validate_gpu_training_config
from mopforge.models import ModelArchitectureConfig, build_tiny_model_from_architecture
from mopforge.tokenization import TokenizerSpec, build_tokenizer
from mopforge.training import (
    TrainableParameterPolicy,
    apply_trainable_policy,
    count_parameters,
)
from scripts.compare_gpu_runs import compare_runs
from tests.test_gpu_trainer import _config


pytest.importorskip("torch")


EFFICIENCY_CONFIGS = [
    "configs/jobs/100m_dense_colab_efficiency.json",
    "configs/jobs/100m_dense_extended_efficiency.json",
    "configs/jobs/100m_mop_full_colab_efficiency.json",
    "configs/jobs/100m_mop_full_extended_efficiency.json",
    "configs/jobs/100m_mop_adapters_only_colab_efficiency.json",
    "configs/jobs/100m_mop_core_frozen_colab_efficiency.json",
    "configs/jobs/100m_mop_router_adapters_colab_efficiency.json",
    "configs/jobs/100m_mop_warm_adapters_efficiency.json",
    "configs/jobs/100m_mop_warm_adapters_norm_head_efficiency.json",
    "configs/jobs/100m_mop_core_frozen_quality_efficiency.json",
    "configs/jobs/100m_mop_warm_adapters_64_colab_efficiency.json",
    "configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json",
    "configs/jobs/100m_mop_core_frozen_quality_colab_efficiency.json",
    "configs/jobs/100m_mop_routed_ffn_expert_efficiency.json",
    "configs/jobs/100m_mop_warm_lora_deltas_efficiency.json",
]


def test_gpu_trainer_records_efficiency_metrics(tmp_path) -> None:
    result = GPUTrainer(_config(tmp_path, name="efficiency_gpu")).train()

    efficiency = result.metrics["efficiency"]
    for key in (
        "tokens_per_sec",
        "samples_per_sec",
        "step_time_sec",
        "total_train_time_sec",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "final_allocated_gb",
        "final_reserved_gb",
        "total_params",
        "trainable_params",
        "frozen_params",
        "trainable_param_ratio",
        "active_param_estimate",
        "active_param_ratio",
        "active_trainable_param_estimate",
        "active_trainable_param_ratio",
        "shared_frozen_params",
        "routed_module_params",
        "active_module_density",
        "active_adapter_density",
        "generated_condition_density",
        "checkpoint_size_mb",
    ):
        assert key in efficiency
    assert efficiency["tokens_per_sec"] is not None
    assert efficiency["samples_per_sec"] is not None
    assert efficiency["total_params"] > 0
    assert efficiency["checkpoint_size_mb"] is not None

    result_payload = json.loads(Path(result.artifacts["gpu_training_result_json"]).read_text(encoding="utf-8"))
    assert "efficiency" in result_payload["metrics"]


def test_gpu_trainer_records_target_loss_and_best_checkpoint(tmp_path) -> None:
    result = GPUTrainer(
        _config(
            tmp_path,
            name="target_loss_gpu",
            save_every_steps=10,
            target_eval_loss=100.0,
        )
    ).train()

    efficiency = result.metrics["efficiency"]

    assert result.metrics["target_eval_loss_reached"] is True
    assert result.metrics["target_eval_loss_value"] == result.metrics["latest_eval_loss"]
    assert result.state["target_eval_loss_reached"] is True
    assert result.state["target_eval_loss_step"] == 1
    assert efficiency["target_eval_loss"] == 100.0
    assert efficiency["target_eval_loss_reached"] is True
    assert efficiency["time_to_target_loss_sec"] is not None
    assert efficiency["tokens_to_target_loss"] == result.metrics["tokens_seen"]
    assert "best_checkpoint_path" in result.artifacts
    assert Path(result.artifacts["best_checkpoint_path"]).exists()


def test_cuda_memory_helpers_are_safe_without_cuda() -> None:
    reset_cuda_peak_memory(None)
    payload = cuda_memory_metrics(None)

    assert payload == {
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "final_allocated_gb": None,
        "final_reserved_gb": None,
        "device_free_gb": None,
        "device_total_gb": None,
        "num_alloc_retries": None,
        "num_ooms": None,
        "inactive_split_gb": None,
        "allocator_cached_gb": None,
        "non_releasable_gb": None,
    }


def test_sparse_mop_policy_modes_freeze_expected_groups() -> None:
    tokenizer = build_tokenizer(TokenizerSpec(tokenizer_type="byte"))
    dense = build_tiny_model_from_architecture(
        ModelArchitectureConfig(
            name="dense",
            model_type="dense",
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
        ),
        tokenizer=tokenizer,
    )
    dense_trainable = count_parameters(dense)["trainable"]
    for mode in ("adapters_only", "core_frozen", "router_adapters_only"):
        model = build_tiny_model_from_architecture(
            ModelArchitectureConfig(
                name=f"mop-{mode}",
                model_type="mop_oracle",
                d_model=32,
                n_layers=1,
                n_heads=2,
                max_seq_len=64,
                use_fast_adapters=True,
                fast_adapter_names=["coding", "debugging"],
            ),
            tokenizer=tokenizer,
        )
        summaries = apply_trainable_policy(
            model,
            TrainableParameterPolicy(mode=mode, train_fast_adapters=True),
        )
        params = count_parameters(model)
        summary_by_name = {item.name: item for item in summaries}
        assert params["trainable"] < dense_trainable
        assert summary_by_name["shared_core"].trainable_params == 0
        assert any(item.frozen_params > 0 for item in summaries)
        if mode == "core_frozen":
            assert any(item.name.startswith("module:") and item.trainable_params > 0 for item in summaries)
        else:
            assert all(not item.name.startswith("module:") or item.trainable_params == 0 for item in summaries)
        assert any(item.name.startswith("adapter:") and item.trainable_params > 0 for item in summaries)


def test_efficiency_config_templates_validate() -> None:
    for path in EFFICIENCY_CONFIGS:
        envelope = MoPForgeConfig.load(path)
        assert envelope.kind == "gpu_train"
        config = gpu_training_config_from_envelope(envelope)
        messages = validate_gpu_training_config(config)
        assert not [message for message in messages if message.startswith("ERROR:")], path


def test_compare_script_handles_old_and_new_run_json(tmp_path) -> None:
    root = tmp_path / "gpu_runs"
    old = _fake_run(root, "old-run", nested=False)
    new = _fake_run(root, "new-run", nested=True)

    rows = compare_runs([old.name, new.name], gpu_runs_dir=root)

    assert len(rows) == 2
    assert rows[0]["tokens_per_sec"] is None
    assert rows[1]["tokens_per_sec"] == 123.4
    assert rows[1]["checkpoint_size_mb"] is not None
    assert rows[1]["distillation_top_k"] == 16
    assert rows[1]["cached_backbone_offloaded_param_count"] == 900
    assert rows[1]["hard_example_replay_enabled"] is True
    assert rows[1]["hard_replayed_example_count"] == 12
    assert rows[1]["best_eval_loss"] == 1.3
    assert rows[1]["target_eval_loss"] == 1.35
    assert rows[1]["target_eval_loss_reached"] is True
    assert rows[1]["time_to_target_loss_sec"] == 12.0
    assert rows[1]["gen_exact_match_rate"] == 0.75
    assert rows[1]["gen_verifier_pass_rate"] == 0.8
    assert rows[1]["gen_syntax_pass_rate"] == 0.9


def test_compare_cli_outputs_json_and_csv(tmp_path, capsys) -> None:
    root = tmp_path / "gpu_runs"
    first = _fake_run(root, "dense", nested=True)
    second = _fake_run(root, "mop", nested=True, trainable_ratio=0.05)
    output = tmp_path / "comparison.json"

    assert cli_main(
        [
            "gpu",
            "compare-runs",
            first.name,
            second.name,
            "--gpu-runs-dir",
            str(root),
            "--output",
            str(output),
        ]
    ) == 0
    stdout = capsys.readouterr().out

    assert "run_id" in stdout
    assert output.exists()
    assert output.with_suffix(".csv").exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 2


def _fake_run(
    root: Path,
    run_id: str,
    *,
    nested: bool,
    trainable_ratio: float = 1.0,
) -> Path:
    run_dir = root / run_id
    checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    model = {
        "parameter_counts": {"total": 1000, "trainable": int(1000 * trainable_ratio), "frozen": 1000 - int(1000 * trainable_ratio)},
        "trainable_param_ratio": trainable_ratio,
        "active_param_estimate": 500,
        "routing_mode": "oracle",
        "active_module_density": 0.5,
        "active_adapter_density": 0.25,
        "generated_condition_density": 0.0,
    }
    metrics = {
        "status": "completed",
        "latest_train_loss": 1.2,
        "latest_eval_loss": 1.4,
        "best_eval_loss": 1.3,
        "target_eval_loss": 1.35,
        "target_eval_loss_reached": True,
        "target_eval_loss_time_sec": 12.0,
        "target_eval_loss_tokens_seen": 4096,
        "target_eval_loss_samples_seen": 64,
        "runtime": {"selected_device": "cpu", "selected_precision": "fp32"},
        "generation_eval": {
            "gen_exact_match_rate": 0.75,
            "gen_verifier_pass_rate": 0.8,
            "gen_syntax_pass_rate": 0.9,
            "gen_compile_pass_rate": 0.9,
        },
        "model": model,
    }
    if nested:
        metrics["efficiency"] = {
            "tokens_per_sec": 123.4,
            "samples_per_sec": 12.3,
            "target_eval_loss": 1.35,
            "target_eval_loss_reached": True,
            "time_to_target_loss_sec": 12.0,
            "tokens_to_target_loss": 4096,
            "samples_to_target_loss": 64,
            "target_peak_allocated_gb": 0.4,
            "target_peak_reserved_gb": 0.5,
            "peak_reserved_gb": None,
            "trainable_param_ratio": trainable_ratio,
            "active_param_ratio": 0.5,
            "checkpoint_size_mb": None,
            "distillation_enabled": True,
            "distillation_weight": 0.2,
            "distillation_top_k": 16,
            "hard_example_replay_enabled": True,
            "hard_example_count": 6,
            "hard_replayed_example_count": 12,
            "cached_backbone_offloaded_param_count": 900,
        }
    result = {
        "run_id": run_id,
        "status": "completed",
        "metrics": metrics,
        "state": {"latest_checkpoint_path": str(checkpoint)},
        "artifacts": {"latest_checkpoint_path": str(checkpoint)},
        "runtime_metadata": {"selected_device": "cpu", "selected_precision": "fp32"},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "gpu_training_result.json").write_text(json.dumps(result), encoding="utf-8")
    return run_dir
