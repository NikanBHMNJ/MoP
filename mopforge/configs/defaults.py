"""Default CPU-safe config templates for MoP-Forge."""

from __future__ import annotations

from mopforge.analysis.config import AnalysisConfig
from mopforge.ablations.config import AblationConfig, AblationVariant
from mopforge.benchmarks import BenchmarkConfig
from mopforge.baselines.config import BaselineConfig
from mopforge.configs.io import MoPForgeConfig
from mopforge.datasets import DatasetConfig
from mopforge.gpu import GPUTrainingConfig
from mopforge.importers import ResultImportConfig
from mopforge.manifests.resources import ResourceSpec
from mopforge.manifests.run_manifest import ManifestConfig
from mopforge.models.architectures import ModelArchitectureConfig
from mopforge.models.manifest import ModelConfig
from mopforge.papers import PaperReportConfig
from mopforge.runtime import RuntimeConfig


def default_trainer_config() -> MoPForgeConfig:
    """Return a tiny oracle-MoP trainer smoke config."""

    return MoPForgeConfig(
        kind="trainer",
        payload={
            "run_name": "cli_tiny_trainer_mop",
            "model_type": "mop_oracle",
            "lesson_path": "data/indexed_lessons.jsonl",
            "index_path": "data/kts_index.sqlite",
            "max_steps": 1,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "eval_batches": 1,
            "batch_size": 2,
            "max_seq_len": 256,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "device": "cpu",
            "save_full_checkpoints": True,
            "checkpoint_every_steps": 1,
        },
        metadata={"description": "Tiny CPU oracle-MoP trainer smoke run"},
    )


def default_sft_config(mode: str = "sft_full") -> MoPForgeConfig:
    """Return a CPU-safe SFT config for ``mode``."""

    payload = _base_sft_payload(mode)
    if mode == "sft_module":
        payload.update({"model_type": "mop_oracle", "target_modules": ["coding"]})
    elif mode == "sft_adapter":
        payload.update(
            {
                "model_type": "mop_oracle",
                "target_modules": ["coding"],
                "use_fast_adapters": True,
                "fast_adapter_names": ["coding", "debugging", "repair"],
            }
        )
    elif mode == "sft_generated":
        payload.update(
            {
                "model_type": "mop_oracle",
                "target_modules": ["coding"],
                "use_generated_params": True,
                "generated_condition_names": ["coding", "debugging", "repair"],
                "generated_rank": 4,
            }
        )
    elif mode != "sft_full":
        raise ValueError(f"Unsupported default SFT mode: {mode}")
    return MoPForgeConfig(
        kind="sft",
        payload=payload,
        metadata={"description": f"Tiny CPU {mode} smoke run"},
    )


def default_pretrain_config() -> MoPForgeConfig:
    """Return a CPU-safe continued-pretraining smoke config."""

    return MoPForgeConfig(
        kind="pretrain",
        payload={
            "run_name": "cli_tiny_continued_pretrain",
            "corpus_path": "data/text_corpus.jsonl",
            "lesson_path": "data/indexed_lessons.jsonl",
            "model_type": "dense",
            "max_steps": 1,
            "eval_batches": 1,
            "batch_size": 1,
            "max_seq_len": 128,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "device": "cpu",
            "save_full_checkpoints": True,
            "checkpoint_every_steps": 1,
        },
        metadata={"description": "Tiny CPU continued-pretraining smoke run"},
    )


def default_generated_sft_config() -> MoPForgeConfig:
    """Return the generated-parameter SFT template."""

    return default_sft_config("sft_generated")


def default_fast_adapter_sft_config() -> MoPForgeConfig:
    """Return the fast-adapter SFT template."""

    return default_sft_config("sft_adapter")


def default_experiment_dense_vs_mop_config() -> MoPForgeConfig:
    """Return a tiny CPU experiment comparing dense and oracle-MoP SFT."""

    dense = default_sft_config("sft_full")
    mop = default_sft_config("sft_module")
    return MoPForgeConfig(
        kind="experiment",
        payload={
            "name": "dense_vs_mop_sft_cpu",
            "kind": "list",
            "description": "Tiny CPU dense full SFT vs oracle MoP module SFT.",
            "runs": [dense.to_dict(), mop.to_dict()],
            "max_runs": 2,
            "tags": ["cpu", "smoke", "sft"],
        },
        metadata={"description": "Tiny CPU dense-vs-MoP experiment"},
    )


def default_experiment_adapter_vs_generated_config() -> MoPForgeConfig:
    """Return a tiny CPU experiment comparing adapters and generated params."""

    adapter = default_sft_config("sft_adapter")
    generated = default_sft_config("sft_generated")
    return MoPForgeConfig(
        kind="experiment",
        payload={
            "name": "adapter_vs_generated_sft_cpu",
            "kind": "list",
            "description": "Tiny CPU fast-adapter SFT vs generated-parameter SFT.",
            "runs": [adapter.to_dict(), generated.to_dict()],
            "max_runs": 2,
            "tags": ["cpu", "smoke", "sft", "adapters"],
        },
        metadata={"description": "Tiny CPU adapter-vs-generated experiment"},
    )


def default_benchmark_loss_config() -> MoPForgeConfig:
    """Return a tiny CPU loss benchmark config."""

    return _benchmark_envelope(
        BenchmarkConfig(
            name="loss_smoke_cpu",
            benchmark_type="loss",
            description="Tiny CPU loss benchmark over KTS lessons.",
            max_examples=4,
            batch_size=2,
            max_seq_len=128,
        )
    )


def default_benchmark_code_correctness_config() -> MoPForgeConfig:
    """Return a tiny CPU generated-code correctness benchmark config."""

    return _benchmark_envelope(
        BenchmarkConfig(
            name="code_correctness_smoke_cpu",
            benchmark_type="code_correctness",
            description="Tiny CPU generated-code correctness smoke benchmark.",
            max_examples=2,
            generation_examples=2,
            generation_max_new_tokens=32,
            max_seq_len=128,
        )
    )


def default_benchmark_router_config() -> MoPForgeConfig:
    """Return a tiny CPU router benchmark config."""

    return _benchmark_envelope(
        BenchmarkConfig(
            name="router_smoke_cpu",
            benchmark_type="router",
            description="Tiny CPU router quality smoke benchmark.",
            max_examples=4,
            batch_size=2,
            max_seq_len=128,
        )
    )


def default_benchmark_parameter_efficiency_config() -> MoPForgeConfig:
    """Return a tiny CPU parameter-efficiency benchmark config."""

    return _benchmark_envelope(
        BenchmarkConfig(
            name="parameter_efficiency_smoke_cpu",
            benchmark_type="parameter_efficiency",
            description="Tiny CPU parameter-count smoke benchmark.",
            model_type="mop_oracle",
            target_modules=["coding"],
            use_fast_adapters=True,
            max_seq_len=128,
            metadata={"trainable_policy_mode": "fast_adapters_only"},
        )
    )


def default_benchmark_composite_config() -> MoPForgeConfig:
    """Return a tiny CPU composite benchmark config."""

    return _benchmark_envelope(
        BenchmarkConfig(
            name="composite_smoke_cpu",
            benchmark_type="composite",
            description="Tiny CPU composite benchmark over parameter/loss/code paths.",
            model_type="mop_oracle",
            target_modules=["coding"],
            use_fast_adapters=True,
            max_examples=3,
            generation_examples=2,
            generation_max_new_tokens=32,
            batch_size=1,
            max_seq_len=128,
            metadata={"trainable_policy_mode": "fast_adapters_only"},
        )
    )


def default_analysis_adapter_vs_generated_config() -> MoPForgeConfig:
    """Return a template analysis for adapter-vs-generated SFT outputs."""

    return _analysis_envelope(
        AnalysisConfig(
            name="adapter_vs_generated_analysis",
            description=(
                "Analyze adapter-vs-generated experiment summaries and benchmark metrics."
            ),
            metrics=[
                "final_eval_loss",
                "eval_loss_mean",
                "trainable_ratio",
                "trainable_params",
                "pass_rate",
            ],
            group_by=["mode"],
            rank_by="final_eval_loss",
            rank_mode="min",
            baseline_filter={"mode": "sft_adapter"},
            metadata={
                "allow_empty_sources": True,
                "source_note": "Add experiment_ids and benchmark_ids from local runs before final reporting.",
            },
        )
    )


def default_analysis_dense_vs_mop_config() -> MoPForgeConfig:
    """Return a template analysis for dense-vs-MoP outputs."""

    return _analysis_envelope(
        AnalysisConfig(
            name="dense_vs_mop_analysis",
            description="Analyze dense-vs-oracle-MoP experiment summaries.",
            metrics=["final_eval_loss", "eval_loss_mean", "trainable_ratio"],
            group_by=["model_type"],
            rank_by="final_eval_loss",
            rank_mode="min",
            baseline_filter={"model_type": "dense"},
            metadata={
                "allow_empty_sources": True,
                "source_note": "Add experiment_ids from local dense-vs-MoP experiments.",
            },
        )
    )


def default_analysis_composite_report_config() -> MoPForgeConfig:
    """Return a template composite local analysis report config."""

    return _analysis_envelope(
        AnalysisConfig(
            name="composite_analysis_report",
            description="Build a local Markdown report from experiment and benchmark artifacts.",
            metrics=[
                "final_eval_loss",
                "eval_loss_mean",
                "pass_rate",
                "router_exact_match_rate",
                "trainable_ratio",
                "trainable_params",
            ],
            group_by=["source_type", "mode"],
            rank_by="eval_loss_mean",
            rank_mode="min",
            metadata={
                "allow_empty_sources": True,
                "source_note": "Use analyze compare with dynamic experiment/benchmark IDs for populated reports.",
            },
        )
    )


def default_dataset_register_lessons_config() -> MoPForgeConfig:
    """Return a local lesson dataset registration template."""

    return _dataset_envelope(
        DatasetConfig(
            action="register",
            name="coding_bugfix",
            dataset_id="coding_bugfix",
            kind="lessons",
            source_paths=["data/coding_bugfix_lessons.jsonl"],
            metadata={"description": "Register local coding bugfix lessons."},
        )
    )


def default_dataset_register_corpus_config() -> MoPForgeConfig:
    """Return a local corpus dataset registration template."""

    return _dataset_envelope(
        DatasetConfig(
            action="register",
            name="text_corpus",
            dataset_id="text_corpus",
            kind="corpus",
            source_paths=["data/text_corpus.jsonl"],
            metadata={"description": "Register local continued-pretraining corpus."},
        )
    )


def default_dataset_split_lessons_config() -> MoPForgeConfig:
    """Return a local lesson dataset split template."""

    return _dataset_envelope(
        DatasetConfig(
            action="split",
            name="coding_bugfix_split",
            dataset_ref="coding_bugfix",
            kind="lessons",
            split_train=0.8,
            split_eval=0.1,
            split_test=0.1,
            split_seed=123,
            metadata={"description": "Create deterministic splits for coding_bugfix."},
        )
    )


def default_model_tiny_dense_config() -> MoPForgeConfig:
    return _model_envelope(ModelArchitectureConfig(name="tiny_dense_base", model_type="dense", d_model=32, n_layers=1, n_heads=2, max_seq_len=128))


def default_model_tiny_mop_config() -> MoPForgeConfig:
    return _model_envelope(ModelArchitectureConfig(name="tiny_mop_oracle", model_type="mop_oracle", d_model=32, n_layers=1, n_heads=2, max_seq_len=128))


def default_model_tiny_adapter_config() -> MoPForgeConfig:
    return _model_envelope(ModelArchitectureConfig(name="tiny_mop_adapter", model_type="mop_oracle", d_model=32, n_layers=1, n_heads=2, max_seq_len=128, use_fast_adapters=True, fast_adapter_names=["coding", "debugging", "repair"]))


def default_model_tiny_generated_config() -> MoPForgeConfig:
    return _model_envelope(ModelArchitectureConfig(name="tiny_mop_generated", model_type="mop_oracle", d_model=32, n_layers=1, n_heads=2, max_seq_len=128, use_generated_params=True, generated_condition_names=["coding", "debugging", "repair"]))


def default_model_future_2b_mop_config() -> MoPForgeConfig:
    return _model_envelope(ModelArchitectureConfig(name="mop_2b_research_config", model_type="future_large", d_model=2048, n_layers=24, n_heads=16, max_seq_len=2048, intended_scale="medium_gpu", metadata={"registry_only": True}))


def default_manifest_cpu_smoke_config() -> MoPForgeConfig:
    return _manifest_envelope(ManifestConfig(name="cpu_smoke_plan", config_payload=default_sft_config("sft_full").to_dict(), resource_spec=ResourceSpec(accelerator="cpu").to_dict()))


def default_manifest_a100_2b_plan_config() -> MoPForgeConfig:
    return _manifest_envelope(ManifestConfig(name="a100_2b_plan", config_payload=default_sft_config("sft_full").to_dict(), resource_spec=ResourceSpec(accelerator="a100_80gb", num_gpus=1, gpu_memory_gb=80, precision="bf16").to_dict()))


def default_manifest_h100_mop_plan_config() -> MoPForgeConfig:
    return _manifest_envelope(ManifestConfig(name="h100_mop_plan", config_payload=default_experiment_adapter_vs_generated_config().to_dict(), resource_spec=ResourceSpec(accelerator="h100_80gb", num_gpus=1, gpu_memory_gb=80, precision="bf16").to_dict()))


def default_manifest_b300_multi_gpu_plan_config() -> MoPForgeConfig:
    return _manifest_envelope(ManifestConfig(name="b300_multi_gpu_plan", config_payload=default_experiment_dense_vs_mop_config().to_dict(), resource_spec=ResourceSpec(accelerator="b300", num_gpus=2, precision="bf16", nodes=1).to_dict()))


def default_import_results_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="import", payload=ResultImportConfig(name="local_results_import", source_path="runs").to_dict(), metadata={"description": "Import local result artifacts."})


def default_ablation_adapter_vs_generated_config() -> MoPForgeConfig:
    return _ablation_envelope(
        AblationConfig(
            name="adapter_vs_generated_ablation",
            description="Tiny CPU adapter vs generated-params ablation.",
            base_config=default_sft_config("sft_adapter").to_dict(),
            variants=[
                AblationVariant("adapter", overrides={"mode": "sft_adapter", "use_fast_adapters": True, "use_generated_params": False}),
                AblationVariant("generated", overrides={"mode": "sft_generated", "use_fast_adapters": False, "use_generated_params": True, "generated_condition_names": ["coding", "debugging", "repair"]}),
            ],
        )
    )


def default_ablation_dense_vs_mop_config() -> MoPForgeConfig:
    return _ablation_envelope(
        AblationConfig(
            name="dense_vs_mop_ablation",
            base_config=default_sft_config("sft_full").to_dict(),
            variants=[
                AblationVariant("dense", overrides={"mode": "sft_full", "model_type": "dense"}),
                AblationVariant("mop", overrides={"mode": "sft_module", "model_type": "mop_oracle", "target_modules": ["coding"]}),
            ],
        )
    )


def default_ablation_fastparam_policy_config() -> MoPForgeConfig:
    return _ablation_envelope(
        AblationConfig(
            name="fastparam_policy_ablation",
            base_config=default_sft_config("sft_adapter").to_dict(),
            variants=[
                AblationVariant("adapter_only", overrides={"mode": "sft_adapter", "trainable_policy_mode": "fast_adapters_only"}),
                AblationVariant("module_only", overrides={"mode": "sft_module", "use_fast_adapters": False, "trainable_policy_mode": "target_modules_only"}),
            ],
        )
    )


def default_baseline_dense_adapter_mop_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="baseline", payload=BaselineConfig().to_dict(), metadata={"description": "Tiny dense/adapter/generated/MoP baseline comparison."})


def default_stats_summary_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="stats", payload={"input_path": "reports/example/normalized_results.json", "group_by": "mode", "metrics": ["final_eval_loss"], "output_root": "outputs/stats"}, metadata={"description": "Statistical summary table template."})


def default_paper_report_smoke_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="paper_report", payload=PaperReportConfig(title="MoP-Forge CPU Smoke Report", abstract="Conservative local report scaffold for CPU-smoke artifacts.").to_dict(), metadata={"description": "Paper-style Markdown scaffold."})


def default_runtime_cpu_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="runtime", payload=RuntimeConfig(device="cpu", precision="fp32").to_dict(), metadata={"description": "CPU fp32 runtime config."})


def default_runtime_auto_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="runtime", payload=RuntimeConfig(device="auto", precision="auto", enable_amp=True, require_device_available=False).to_dict(), metadata={"description": "Auto device/precision runtime config."})


def default_runtime_cuda_bf16_plan_config() -> MoPForgeConfig:
    return MoPForgeConfig(kind="runtime", payload=RuntimeConfig(device="cuda", precision="bf16", enable_amp=True, allow_tf32=True, require_device_available=False, metadata={"plan_only": True}).to_dict(), metadata={"description": "CUDA bf16 runtime plan; falls back on CPU-only machines."})


def default_trainer_runtime_auto_config() -> MoPForgeConfig:
    envelope = default_trainer_config()
    envelope.payload.update({"device": "auto", "precision": "auto", "enable_amp": True, "require_device_available": False})
    envelope.metadata["description"] = "Tiny trainer runtime-auto smoke config."
    return envelope


def default_sft_runtime_auto_config() -> MoPForgeConfig:
    envelope = default_sft_config("sft_full")
    envelope.payload.update({"device": "auto", "precision": "auto", "enable_amp": True, "require_device_available": False})
    envelope.metadata["description"] = "Tiny SFT runtime-auto smoke config."
    return envelope


def default_benchmark_runtime_auto_config() -> MoPForgeConfig:
    envelope = default_benchmark_loss_config()
    envelope.payload.update({"device": "auto", "precision": "auto", "enable_amp": True, "require_device_available": False})
    envelope.metadata["description"] = "Tiny benchmark runtime-auto smoke config."
    return envelope


def default_gpu_tiny_smoke_config() -> MoPForgeConfig:
    return _gpu_envelope(
        GPUTrainingConfig(
            name="tiny_gpu_smoke",
            model_type="mop_oracle",
            max_steps=2,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            eval_every_steps=1,
            eval_batches=1,
            save_every_steps=1,
            log_every_steps=1,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=128,
            target_modules=["coding"],
            device="auto",
            precision="auto",
            enable_amp=True,
            allow_tf32=True,
            require_device_available=False,
            max_train_examples=8,
            max_eval_examples=4,
            metadata={"target_gpu_memory_gb": 8, "description": "Tiny CPU/CUDA fallback GPU smoke."},
        )
    )


def default_gpu_100m_mop_a100_config() -> MoPForgeConfig:
    return _gpu_envelope(
        GPUTrainingConfig(
            name="100m_mop_a100_smoke",
            model_type="mop_oracle",
            max_steps=100,
            micro_batch_size=1,
            gradient_accumulation_steps=8,
            d_model=768,
            n_layers=12,
            n_heads=12,
            max_seq_len=1024,
            target_modules=["coding", "debugging", "repair"],
            use_fast_adapters=True,
            fast_adapter_names=["coding", "debugging", "repair"],
            device="cuda",
            precision="bf16",
            enable_amp=True,
            allow_tf32=True,
            require_device_available=False,
            activation_checkpointing=True,
            metadata={"parameter_count": 100_000_000, "target_gpu_memory_gb": 80, "profile": "a100_100m_smoke"},
        )
    )


def default_gpu_500m_mop_h100_config() -> MoPForgeConfig:
    return _gpu_envelope(_large_gpu_profile("500m_mop_h100", 500_000_000, 1024, 24, 16, 2048, 80, plan_only=True))


def default_gpu_1b_mop_h100_config() -> MoPForgeConfig:
    return _gpu_envelope(_large_gpu_profile("1b_mop_h100_bf16", 1_000_000_000, 1536, 28, 16, 2048, 80, plan_only=True))


def default_gpu_2b_mop_a100_plan_config() -> MoPForgeConfig:
    return _gpu_envelope(_large_gpu_profile("2b_mop_a100_plan", 2_000_000_000, 2048, 32, 16, 2048, 80, plan_only=True))


def default_gpu_7b_mop_h100_plan_config() -> MoPForgeConfig:
    return _gpu_envelope(_large_gpu_profile("7b_mop_h100_plan", 7_000_000_000, 4096, 32, 32, 4096, 80, plan_only=True))


def default_gpu_multigpu_torchrun_plan_config() -> MoPForgeConfig:
    envelope = _gpu_envelope(_large_gpu_profile("multigpu_mop_torchrun_plan", 1_000_000_000, 1536, 28, 16, 2048, 80, plan_only=True))
    envelope.metadata["distributed"] = {
        "strategy": "torchrun",
        "num_nodes": 1,
        "nproc_per_node": 2,
        "backend": "nccl",
        "dry_run": True,
    }
    envelope.metadata["description"] = "Torchrun dry-run plan for multi-GPU MoP training."
    return envelope


def list_default_config_names() -> list[str]:
    """Return supported default template names."""

    return [
        "trainer",
        "sft_full",
        "sft_module",
        "sft_adapter",
        "sft_generated",
        "pretrain",
        "generated_sft",
        "fast_adapter_sft",
        "experiment_dense_vs_mop",
        "experiment_adapter_vs_generated",
        "benchmark_loss",
        "benchmark_code_correctness",
        "benchmark_router",
        "benchmark_parameter_efficiency",
        "benchmark_composite",
        "analysis_adapter_vs_generated",
        "analysis_dense_vs_mop",
        "analysis_composite_report",
        "dataset_register_lessons",
        "dataset_register_corpus",
        "dataset_split_lessons",
        "model_tiny_dense",
        "model_tiny_mop",
        "model_tiny_adapter",
        "model_tiny_generated",
        "model_future_2b_mop",
        "manifest_cpu_smoke",
        "manifest_a100_2b_plan",
        "manifest_h100_mop_plan",
        "manifest_b300_multi_gpu_plan",
        "import_results",
        "ablation_dense_vs_mop",
        "ablation_adapter_vs_generated",
        "ablation_fastparam_policy",
        "baseline_dense_adapter_mop",
        "stats_summary",
        "paper_report_smoke",
        "runtime_cpu",
        "runtime_auto",
        "runtime_cuda_bf16_plan",
        "trainer_runtime_auto",
        "sft_runtime_auto",
        "benchmark_runtime_auto",
        "gpu_tiny_smoke",
        "gpu_100m_mop_a100",
        "gpu_500m_mop_h100",
        "gpu_1b_mop_h100",
        "gpu_2b_mop_a100_plan",
        "gpu_7b_mop_h100_plan",
        "gpu_multigpu_torchrun_plan",
    ]


def get_default_config(name: str) -> MoPForgeConfig:
    """Return a named default config template."""

    if name == "trainer":
        return default_trainer_config()
    if name == "pretrain":
        return default_pretrain_config()
    if name == "generated_sft":
        return default_generated_sft_config()
    if name == "fast_adapter_sft":
        return default_fast_adapter_sft_config()
    if name == "experiment_dense_vs_mop":
        return default_experiment_dense_vs_mop_config()
    if name == "experiment_adapter_vs_generated":
        return default_experiment_adapter_vs_generated_config()
    if name == "benchmark_loss":
        return default_benchmark_loss_config()
    if name == "benchmark_code_correctness":
        return default_benchmark_code_correctness_config()
    if name == "benchmark_router":
        return default_benchmark_router_config()
    if name == "benchmark_parameter_efficiency":
        return default_benchmark_parameter_efficiency_config()
    if name == "benchmark_composite":
        return default_benchmark_composite_config()
    if name == "analysis_adapter_vs_generated":
        return default_analysis_adapter_vs_generated_config()
    if name == "analysis_dense_vs_mop":
        return default_analysis_dense_vs_mop_config()
    if name == "analysis_composite_report":
        return default_analysis_composite_report_config()
    if name == "dataset_register_lessons":
        return default_dataset_register_lessons_config()
    if name == "dataset_register_corpus":
        return default_dataset_register_corpus_config()
    if name == "dataset_split_lessons":
        return default_dataset_split_lessons_config()
    if name == "model_tiny_dense":
        return default_model_tiny_dense_config()
    if name == "model_tiny_mop":
        return default_model_tiny_mop_config()
    if name == "model_tiny_adapter":
        return default_model_tiny_adapter_config()
    if name == "model_tiny_generated":
        return default_model_tiny_generated_config()
    if name == "model_future_2b_mop":
        return default_model_future_2b_mop_config()
    if name == "manifest_cpu_smoke":
        return default_manifest_cpu_smoke_config()
    if name == "manifest_a100_2b_plan":
        return default_manifest_a100_2b_plan_config()
    if name == "manifest_h100_mop_plan":
        return default_manifest_h100_mop_plan_config()
    if name == "manifest_b300_multi_gpu_plan":
        return default_manifest_b300_multi_gpu_plan_config()
    if name == "import_results":
        return default_import_results_config()
    if name == "ablation_dense_vs_mop":
        return default_ablation_dense_vs_mop_config()
    if name == "ablation_adapter_vs_generated":
        return default_ablation_adapter_vs_generated_config()
    if name == "ablation_fastparam_policy":
        return default_ablation_fastparam_policy_config()
    if name == "baseline_dense_adapter_mop":
        return default_baseline_dense_adapter_mop_config()
    if name == "stats_summary":
        return default_stats_summary_config()
    if name == "paper_report_smoke":
        return default_paper_report_smoke_config()
    if name == "runtime_cpu":
        return default_runtime_cpu_config()
    if name == "runtime_auto":
        return default_runtime_auto_config()
    if name == "runtime_cuda_bf16_plan":
        return default_runtime_cuda_bf16_plan_config()
    if name == "trainer_runtime_auto":
        return default_trainer_runtime_auto_config()
    if name == "sft_runtime_auto":
        return default_sft_runtime_auto_config()
    if name == "benchmark_runtime_auto":
        return default_benchmark_runtime_auto_config()
    if name == "gpu_tiny_smoke":
        return default_gpu_tiny_smoke_config()
    if name == "gpu_100m_mop_a100":
        return default_gpu_100m_mop_a100_config()
    if name == "gpu_500m_mop_h100":
        return default_gpu_500m_mop_h100_config()
    if name == "gpu_1b_mop_h100":
        return default_gpu_1b_mop_h100_config()
    if name == "gpu_2b_mop_a100_plan":
        return default_gpu_2b_mop_a100_plan_config()
    if name == "gpu_7b_mop_h100_plan":
        return default_gpu_7b_mop_h100_plan_config()
    if name == "gpu_multigpu_torchrun_plan":
        return default_gpu_multigpu_torchrun_plan_config()
    if name in {"sft_full", "sft_module", "sft_adapter", "sft_generated"}:
        return default_sft_config(name)
    valid = ", ".join(list_default_config_names())
    raise ValueError(f"Unknown default config {name!r}. Valid names: {valid}.")


def _base_sft_payload(mode: str) -> dict:
    return {
        "mode": mode,
        "model_type": "dense",
        "lesson_path": "data/indexed_lessons.jsonl",
        "index_path": "data/kts_index.sqlite",
        "max_steps": 1,
        "eval_batches": 1,
        "batch_size": 2,
        "max_seq_len": 256,
        "learning_rate": 1e-3,
        "save_checkpoints": True,
        "save_full_checkpoints": True,
        "checkpoint_every_steps": 1,
    }


def _benchmark_envelope(config: BenchmarkConfig) -> MoPForgeConfig:
    return MoPForgeConfig(
        kind="benchmark",
        payload=config.to_dict(),
        metadata={"description": config.description or f"Tiny CPU {config.benchmark_type} benchmark"},
    )


def _analysis_envelope(config: AnalysisConfig) -> MoPForgeConfig:
    return MoPForgeConfig(
        kind="analysis",
        payload=config.to_dict(),
        metadata={"description": config.description or "Local analysis report template"},
    )


def _dataset_envelope(config: DatasetConfig) -> MoPForgeConfig:
    return MoPForgeConfig(
        kind="dataset",
        payload=config.to_dict(),
        metadata={"description": config.metadata.get("description", "Local dataset registry action")},
    )


def _model_envelope(architecture: ModelArchitectureConfig) -> MoPForgeConfig:
    return MoPForgeConfig(
        kind="model",
        payload=ModelConfig(name=architecture.name, model_id=architecture.name, architecture=architecture.to_dict()).to_dict(),
        metadata={"description": f"Register model architecture {architecture.name}"},
    )


def _manifest_envelope(config: ManifestConfig) -> MoPForgeConfig:
    return MoPForgeConfig(kind="manifest", payload=config.to_dict(), metadata={"description": "Research run manifest plan"})


def _ablation_envelope(config: AblationConfig) -> MoPForgeConfig:
    return MoPForgeConfig(kind="ablation", payload=config.to_dict(), metadata={"description": config.description or "Local ablation config"})


def _gpu_envelope(config: GPUTrainingConfig) -> MoPForgeConfig:
    return MoPForgeConfig(
        kind="gpu_train",
        payload=config.to_dict(),
        metadata={"description": config.metadata.get("description", "GPU training/job profile")},
    )


def _large_gpu_profile(
    name: str,
    parameter_count: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    target_gpu_memory_gb: int,
    *,
    plan_only: bool = False,
) -> GPUTrainingConfig:
    return GPUTrainingConfig(
        name=name,
        model_type="mop_oracle",
        max_steps=200,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        eval_every_steps=50,
        eval_batches=2,
        save_every_steps=100,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        target_modules=["coding", "debugging", "repair", "math"],
        use_fast_adapters=True,
        fast_adapter_names=["coding", "debugging", "repair", "math"],
        use_generated_params=True,
        generated_condition_names=["coding", "debugging", "repair", "math"],
        device="cuda",
        precision="bf16",
        enable_amp=True,
        allow_tf32=True,
        require_device_available=False,
        activation_checkpointing=True,
        efficient_attention="auto",
        metadata={
            "parameter_count": parameter_count,
            "target_gpu_memory_gb": target_gpu_memory_gb,
            "plan_only": plan_only,
            "profile": name,
        },
    )
