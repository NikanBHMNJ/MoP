"""Argparse CLI entrypoint for MoP-Forge."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Sequence

import mopforge
from mopforge.ablations import AblationRegistry
from mopforge.analysis import AnalysisConfig, AnalysisRegistry, run_analysis
from mopforge.ablations import AblationConfig, dry_run_ablation, run_ablation
from mopforge.benchmarks import BenchmarkRegistry, run_benchmark
from mopforge.baselines import (
    build_baseline_experiment_config,
    get_baseline,
    list_baselines,
)
from mopforge.configs import (
    MoPForgeConfig,
    ablation_config_from_envelope,
    analysis_config_from_envelope,
    benchmark_config_from_envelope,
    baseline_config_from_envelope,
    dry_run_config,
    experiment_config_from_envelope,
    finetune_config_from_envelope,
    get_default_config,
    gpu_training_config_from_envelope,
    import_config_from_envelope,
    list_default_config_names,
    manifest_config_from_envelope,
    model_config_from_envelope,
    paper_report_config_from_envelope,
    pretrain_config_from_envelope,
    trainer_config_from_envelope,
    validate_config_envelope,
)
from mopforge.experiments import ExperimentRegistry, run_experiment
from mopforge.gpu import (
    DistributedConfig,
    GPUTrainer,
    GPURunRegistry,
    build_torchrun_command,
    config_hash,
    dry_run_gpu_training_config,
    estimate_from_config,
    evaluate_efficiency_gates,
    file_sha256,
    prepare_efficiency_dataset,
    validate_gpu_training_config,
    write_activation_cache,
    write_gate_report,
    write_warm_sparse_sweep_configs,
)
from mopforge.datasets import (
    DatasetRegistry,
    create_dataset_split,
    load_dataset_split,
    write_split_jsonl,
)
from mopforge.importers import ResultImportConfig, ResultImportRegistry, import_results
from mopforge.manifests import (
    ManifestRegistry,
    ResourceSpec,
    command_text,
    config_from_path_or_payload,
    dry_run_payload,
    plan_run_manifest,
)
from mopforge.models import ModelArchitectureConfig, ModelRegistry
from mopforge.papers import PaperReportRegistry, build_paper_report
from mopforge.statistics import make_metric_table, write_table_csv, write_table_json, write_table_markdown
from mopforge.lifecycle.checkpoint import load_full_training_checkpoint
from mopforge.lifecycle.resume import resolve_full_checkpoint_reference
from mopforge.pretrain import ContinuedPretrainConfig
from mopforge.pretrain import run_continued_pretraining
from mopforge.runtime import RuntimeConfig, build_runtime_context, detect_devices, runtime_metadata
from mopforge.sft import FinetuneConfig, list_training_modes, run_finetune
from mopforge.training import TinyTrainer, TrainerConfig


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MoP-Forge CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc()
        else:
            print(f"ERROR: {_format_cli_exception(exc)}")
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mopforge",
        description="MoP-Forge local-first research CLI.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show full Python tracebacks for debugging command failures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="print package version")
    version_parser.set_defaults(func=_cmd_version)

    doctor_parser = subparsers.add_parser("doctor", help="check local MoP-Forge environment")
    doctor_parser.add_argument("--root", default=".", help="workspace root for writable-directory checks")
    doctor_parser.add_argument("--json", action="store_true", help="print machine-readable diagnostics")
    doctor_parser.set_defaults(func=_cmd_doctor)

    modes_parser = subparsers.add_parser("modes", help="training mode commands")
    modes_subparsers = modes_parser.add_subparsers(dest="modes_command", required=True)
    modes_list_parser = modes_subparsers.add_parser("list", help="list FT/SFT modes")
    modes_list_parser.set_defaults(func=_cmd_modes_list)

    config_parser = subparsers.add_parser("config", help="config file commands")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    write_parser = config_subparsers.add_parser(
        "write-default",
        help="write a default config template",
    )
    write_parser.add_argument("name", choices=list_default_config_names())
    write_parser.add_argument("path")
    write_parser.set_defaults(func=_cmd_config_write_default)

    validate_parser = config_subparsers.add_parser("validate", help="validate a config")
    validate_parser.add_argument("path")
    validate_parser.set_defaults(func=_cmd_config_validate)

    dry_run_parser = config_subparsers.add_parser("dry-run", help="resolve a config")
    dry_run_parser.add_argument("path")
    dry_run_parser.set_defaults(func=_cmd_config_dry_run)

    runtime_parser = subparsers.add_parser("runtime", help="runtime/device commands")
    runtime_sub = runtime_parser.add_subparsers(dest="runtime_command", required=True)
    runtime_detect = runtime_sub.add_parser("detect", help="detect runtime devices")
    runtime_detect.add_argument("--json", action="store_true")
    runtime_detect.set_defaults(func=_cmd_runtime_detect)
    runtime_dry = runtime_sub.add_parser("dry-run", help="resolve runtime config")
    runtime_dry.add_argument("--device", default="cpu")
    runtime_dry.add_argument("--precision", default="fp32")
    runtime_dry.add_argument("--enable-amp", action="store_true")
    runtime_dry.add_argument("--allow-tf32", action="store_true")
    runtime_dry.add_argument("--deterministic", action="store_true")
    runtime_dry.add_argument("--compile-model", action="store_true")
    runtime_dry.add_argument("--require-available", dest="require_available", action=argparse.BooleanOptionalAction, default=True)
    runtime_dry.add_argument("--json", action="store_true")
    runtime_dry.set_defaults(func=_cmd_runtime_dry_run)

    gpu_parser = subparsers.add_parser("gpu", help="GPU research beta commands")
    gpu_sub = gpu_parser.add_subparsers(dest="gpu_command", required=True)
    gpu_validate = gpu_sub.add_parser("validate", help="validate a GPU training config without executing it")
    gpu_validate.add_argument("path")
    gpu_validate.set_defaults(func=_cmd_gpu_validate)
    gpu_estimate = gpu_sub.add_parser("estimate", help="estimate GPU training memory")
    gpu_estimate.add_argument("path")
    gpu_estimate.set_defaults(func=_cmd_gpu_estimate)
    gpu_train = gpu_sub.add_parser("train", help="execute a single-device GPU-aware train job")
    gpu_train.add_argument("path")
    gpu_train.add_argument("--device")
    gpu_train.add_argument("--precision")
    gpu_train.add_argument("--allow-plan-run", action="store_true")
    gpu_train.set_defaults(func=_cmd_gpu_train)
    gpu_resume = gpu_sub.add_parser("resume", help="resume a GPU run from run ID or checkpoint")
    gpu_resume.add_argument("checkpoint_or_run_id")
    gpu_resume.set_defaults(func=_cmd_gpu_resume)
    gpu_benchmark = gpu_sub.add_parser("benchmark", help="write a GPU run benchmark scaffold")
    gpu_benchmark.add_argument("run_id")
    gpu_benchmark.set_defaults(func=_cmd_gpu_benchmark)
    gpu_compare = gpu_sub.add_parser("compare-runs", help="compare GPU run efficiency metrics")
    gpu_compare.add_argument("run_ids", nargs="+")
    gpu_compare.add_argument("--gpu-runs-dir", default="gpu_runs")
    gpu_compare.add_argument("--output", default="outputs/gpu_efficiency_comparison.json")
    gpu_compare.add_argument("--output-csv")
    gpu_compare.set_defaults(func=_cmd_gpu_compare_runs)
    gpu_gate = gpu_sub.add_parser(
        "gate-efficiency",
        help="evaluate acceptance gates for a sparse GPU efficiency claim",
    )
    gpu_gate.add_argument("--dense-run", required=True)
    gpu_gate.add_argument("--sparse-run", required=True)
    gpu_gate.add_argument("--gpu-runs-dir", default="gpu_runs")
    gpu_gate.add_argument("--output", default="outputs/gpu_efficiency_gate_report.json")
    gpu_gate.add_argument("--adapter-baseline-eval-loss", type=float, default=5.165306329727173)
    gpu_gate.add_argument("--same-quality-eval-delta", type=float, default=0.25)
    gpu_gate.add_argument("--generation-pass-delta", type=float, default=0.05)
    gpu_gate.add_argument("--vram-target-gb", type=float)
    gpu_gate.set_defaults(func=_cmd_gpu_gate_efficiency)
    gpu_cache = gpu_sub.add_parser(
        "cache-activations",
        help="write frozen-prefix activation cache for sparse tail training",
    )
    gpu_cache.add_argument("path")
    gpu_cache.add_argument("--checkpoint", required=True)
    gpu_cache.add_argument("--output", required=True)
    gpu_cache.add_argument("--max-batches", type=int)
    gpu_cache.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    gpu_cache.set_defaults(func=_cmd_gpu_cache_activations)
    gpu_sweep = gpu_sub.add_parser(
        "write-warm-sparse-sweep",
        help="write warm sparse 64/128/256 bottleneck and LR sweep configs",
    )
    gpu_sweep.add_argument("--base-checkpoint", required=True)
    gpu_sweep.add_argument("--output-dir", default="configs/jobs/warm_sparse_sweep")
    gpu_sweep.add_argument("--activation-cache-path")
    gpu_sweep.add_argument("--dataset-ref")
    gpu_sweep.add_argument("--dataset-split-id")
    gpu_sweep.add_argument("--bottlenecks", type=int, nargs="+", default=[64, 128, 256])
    gpu_sweep.add_argument("--learning-rates", type=float, nargs="+", default=[3e-4, 1e-3, 2e-3])
    gpu_sweep.add_argument("--lora-ranks", type=int, nargs="+", default=[4, 8, 16])
    gpu_sweep.add_argument("--max-steps", type=int, default=2000)
    gpu_sweep.add_argument("--seed", type=int, default=42)
    gpu_sweep.set_defaults(func=_cmd_gpu_write_warm_sparse_sweep)
    gpu_data = gpu_sub.add_parser(
        "prepare-efficiency-data",
        help="generate and register a larger fixed-split coding bugfix dataset",
    )
    gpu_data.add_argument("--source-path", default="data/coding_bugfix_efficiency_lessons.jsonl")
    gpu_data.add_argument("--dataset-root", default="datasets")
    gpu_data.add_argument("--dataset-id", default="coding_bugfix_efficiency")
    gpu_data.add_argument("--count-per-category", type=int, default=100)
    gpu_data.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    gpu_data.add_argument("--timeout-seconds", type=int, default=5)
    gpu_data.add_argument("--split-seed", type=int, default=42)
    gpu_data.add_argument("--train-ratio", type=float, default=0.8)
    gpu_data.add_argument("--eval-ratio", type=float, default=0.1)
    gpu_data.add_argument("--test-ratio", type=float, default=0.1)
    gpu_data.add_argument("--overwrite", action="store_true")
    gpu_data.set_defaults(func=_cmd_gpu_prepare_efficiency_data)
    gpu_launch = gpu_sub.add_parser("launch-torchrun", help="print a torchrun dry-run command; never launches")
    gpu_launch.add_argument("path")
    gpu_launch.add_argument("--dry-run", action="store_true", default=True)
    gpu_launch.set_defaults(func=_cmd_gpu_launch_torchrun)
    gpu_list = gpu_sub.add_parser("list", help="list local GPU runs")
    gpu_list.add_argument("--root", default="gpu_runs")
    gpu_list.set_defaults(func=_cmd_gpu_list)
    gpu_show = gpu_sub.add_parser("show", help="show a GPU run record")
    gpu_show.add_argument("run_id")
    gpu_show.add_argument("--root", default="gpu_runs")
    gpu_show.set_defaults(func=_cmd_gpu_show)

    model_parser = subparsers.add_parser("model", help="model registry commands")
    model_sub = model_parser.add_subparsers(dest="model_command", required=True)
    model_register = model_sub.add_parser("register", help="register model architecture")
    model_register.add_argument("path")
    model_register.add_argument("--root", default="models")
    model_register.set_defaults(func=_cmd_model_register)
    model_list = model_sub.add_parser("list", help="list models")
    model_list.add_argument("--root", default="models")
    model_list.set_defaults(func=_cmd_model_list)
    model_show = model_sub.add_parser("show", help="show model manifest")
    model_show.add_argument("model_ref")
    model_show.add_argument("--root", default="models")
    model_show.set_defaults(func=_cmd_model_show)
    model_versions = model_sub.add_parser("versions", help="list model versions")
    model_versions.add_argument("model_id")
    model_versions.add_argument("--root", default="models")
    model_versions.set_defaults(func=_cmd_model_versions)
    model_snapshot = model_sub.add_parser("snapshot", help="snapshot model")
    model_snapshot.add_argument("model_id")
    model_snapshot.add_argument("--root", default="models")
    model_snapshot.set_defaults(func=_cmd_model_snapshot)

    manifest_parser = subparsers.add_parser("manifest", help="research run manifest commands")
    manifest_sub = manifest_parser.add_subparsers(dest="manifest_command", required=True)
    manifest_create = manifest_sub.add_parser("create", help="create run manifest")
    manifest_create.add_argument("config_path")
    manifest_create.add_argument("--name")
    manifest_create.add_argument("--accelerator", default="cpu")
    manifest_create.add_argument("--num-gpus", type=int, default=0)
    manifest_create.add_argument("--precision", default="fp32")
    manifest_create.add_argument("--root", default="manifests")
    manifest_create.set_defaults(func=_cmd_manifest_create)
    manifest_dry = manifest_sub.add_parser("dry-run", help="show manifest dry run")
    manifest_dry.add_argument("manifest_id")
    manifest_dry.add_argument("--root", default="manifests")
    manifest_dry.set_defaults(func=_cmd_manifest_dry_run)
    manifest_list = manifest_sub.add_parser("list", help="list manifests")
    manifest_list.add_argument("--root", default="manifests")
    manifest_list.set_defaults(func=_cmd_manifest_list)
    manifest_show = manifest_sub.add_parser("show", help="show manifest")
    manifest_show.add_argument("manifest_id")
    manifest_show.add_argument("--root", default="manifests")
    manifest_show.set_defaults(func=_cmd_manifest_show)
    manifest_export = manifest_sub.add_parser("export-command", help="export manifest command")
    manifest_export.add_argument("manifest_id")
    manifest_export.add_argument("--root", default="manifests")
    manifest_export.set_defaults(func=_cmd_manifest_export_command)

    import_parser = subparsers.add_parser("import", help="local result import commands")
    import_sub = import_parser.add_subparsers(dest="import_command", required=True)
    import_results_parser = import_sub.add_parser("results", help="import local result files")
    import_results_parser.add_argument("path")
    import_results_parser.add_argument("--name", required=True)
    import_results_parser.add_argument("--root", default="imports")
    import_results_parser.set_defaults(func=_cmd_import_results)
    import_list = import_sub.add_parser("list", help="list imports")
    import_list.add_argument("--root", default="imports")
    import_list.set_defaults(func=_cmd_import_list)
    import_show = import_sub.add_parser("show", help="show import")
    import_show.add_argument("import_id")
    import_show.add_argument("--root", default="imports")
    import_show.set_defaults(func=_cmd_import_show)

    ablation_parser = subparsers.add_parser("ablation", help="ablation commands")
    ablation_sub = ablation_parser.add_subparsers(dest="ablation_command", required=True)
    ablation_run = ablation_sub.add_parser("run", help="run ablation")
    ablation_run.add_argument("path")
    ablation_run.set_defaults(func=_cmd_ablation_run)
    ablation_dry = ablation_sub.add_parser("dry-run", help="dry-run ablation")
    ablation_dry.add_argument("path")
    ablation_dry.set_defaults(func=_cmd_ablation_dry_run)
    ablation_list = ablation_sub.add_parser("list", help="list ablations")
    ablation_list.add_argument("--root", default="ablations")
    ablation_list.set_defaults(func=_cmd_ablation_list)
    ablation_show = ablation_sub.add_parser("show", help="show ablation")
    ablation_show.add_argument("ablation_id")
    ablation_show.add_argument("--root", default="ablations")
    ablation_show.set_defaults(func=_cmd_ablation_show)

    baseline_parser = subparsers.add_parser("baseline", help="baseline catalog commands")
    baseline_sub = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    baseline_list = baseline_sub.add_parser("list", help="list baselines")
    baseline_list.set_defaults(func=_cmd_baseline_list)
    baseline_show = baseline_sub.add_parser("show", help="show baseline")
    baseline_show.add_argument("name")
    baseline_show.set_defaults(func=_cmd_baseline_show)
    baseline_exp = baseline_sub.add_parser("experiment", help="build baseline experiment config")
    baseline_exp.add_argument("--baselines", nargs="+", required=True)
    baseline_exp.set_defaults(func=_cmd_baseline_experiment)

    stats_parser = subparsers.add_parser("stats", help="statistical table commands")
    stats_sub = stats_parser.add_subparsers(dest="stats_command", required=True)
    stats_sum = stats_sub.add_parser("summarize", help="summarize normalized rows")
    stats_sum.add_argument("path")
    stats_sum.add_argument("--group-by", required=True)
    stats_sum.add_argument("--metric", action="append", required=True)
    stats_sum.add_argument("--output-root", default="outputs/stats")
    stats_sum.set_defaults(func=_cmd_stats_summarize)

    paper_parser = subparsers.add_parser("paper", help="paper-style report commands")
    paper_sub = paper_parser.add_subparsers(dest="paper_command", required=True)
    paper_build = paper_sub.add_parser("build", help="build paper report")
    paper_build.add_argument("path")
    paper_build.set_defaults(func=_cmd_paper_build)
    paper_list = paper_sub.add_parser("list", help="list paper reports")
    paper_list.add_argument("--root", default="paper_reports")
    paper_list.set_defaults(func=_cmd_paper_list)
    paper_show = paper_sub.add_parser("show", help="show paper report")
    paper_show.add_argument("paper_report_id")
    paper_show.add_argument("--root", default="paper_reports")
    paper_show.set_defaults(func=_cmd_paper_show)

    benchmark_parser = subparsers.add_parser("benchmark", help="benchmark commands")
    benchmark_subparsers = benchmark_parser.add_subparsers(
        dest="benchmark_command",
        required=True,
    )
    benchmark_run_parser = benchmark_subparsers.add_parser(
        "run",
        help="run a local benchmark",
    )
    benchmark_run_parser.add_argument("path")
    benchmark_run_parser.add_argument("--registry-root", default="benchmarks")
    benchmark_run_parser.set_defaults(func=_cmd_benchmark_run)

    benchmark_dry_run_parser = benchmark_subparsers.add_parser(
        "dry-run",
        help="summarize a benchmark config",
    )
    benchmark_dry_run_parser.add_argument("path")
    benchmark_dry_run_parser.set_defaults(func=_cmd_benchmark_dry_run)

    benchmark_list_parser = benchmark_subparsers.add_parser(
        "list",
        help="list local benchmarks",
    )
    benchmark_list_parser.add_argument("--registry-root", default="benchmarks")
    benchmark_list_parser.set_defaults(func=_cmd_benchmark_list)

    benchmark_show_parser = benchmark_subparsers.add_parser(
        "show",
        help="show one benchmark record",
    )
    benchmark_show_parser.add_argument("benchmark_id")
    benchmark_show_parser.add_argument("--registry-root", default="benchmarks")
    benchmark_show_parser.set_defaults(func=_cmd_benchmark_show)

    dataset_parser = subparsers.add_parser("dataset", help="dataset registry commands")
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command",
        required=True,
    )
    dataset_register_parser = dataset_subparsers.add_parser(
        "register",
        help="register a local dataset and create a version manifest",
    )
    dataset_register_parser.add_argument("source_paths", nargs="+")
    dataset_register_parser.add_argument("--name", required=True)
    dataset_register_parser.add_argument("--kind", default="lessons")
    dataset_register_parser.add_argument("--dataset-id")
    dataset_register_parser.add_argument("--description", default="")
    dataset_register_parser.add_argument("--tag", action="append", default=[])
    dataset_register_parser.add_argument("--copy-files", action="store_true")
    dataset_register_parser.add_argument("--root", default="datasets")
    dataset_register_parser.set_defaults(func=_cmd_dataset_register)

    dataset_snapshot_parser = dataset_subparsers.add_parser(
        "snapshot",
        help="create a new dataset version manifest",
    )
    dataset_snapshot_parser.add_argument("dataset_id")
    dataset_snapshot_parser.add_argument("--source-path", action="append", default=None)
    dataset_snapshot_parser.add_argument("--copy-files", action="store_true")
    dataset_snapshot_parser.add_argument("--root", default="datasets")
    dataset_snapshot_parser.set_defaults(func=_cmd_dataset_snapshot)

    dataset_split_parser = dataset_subparsers.add_parser(
        "split",
        help="create a deterministic train/eval/test split",
    )
    dataset_split_parser.add_argument("dataset_id")
    dataset_split_parser.add_argument("--train", type=float, default=0.8)
    dataset_split_parser.add_argument("--eval", type=float, default=0.1)
    dataset_split_parser.add_argument("--test", type=float, default=0.1)
    dataset_split_parser.add_argument("--seed", type=int, default=123)
    dataset_split_parser.add_argument("--stratify-by")
    dataset_split_parser.add_argument("--version-id")
    dataset_split_parser.add_argument("--root", default="datasets")
    dataset_split_parser.set_defaults(func=_cmd_dataset_split)

    dataset_list_parser = dataset_subparsers.add_parser(
        "list",
        help="list local datasets",
    )
    dataset_list_parser.add_argument("--root", default="datasets")
    dataset_list_parser.set_defaults(func=_cmd_dataset_list)

    dataset_show_parser = dataset_subparsers.add_parser(
        "show",
        help="show one dataset record or version",
    )
    dataset_show_parser.add_argument("dataset_ref")
    dataset_show_parser.add_argument("--root", default="datasets")
    dataset_show_parser.set_defaults(func=_cmd_dataset_show)

    dataset_versions_parser = dataset_subparsers.add_parser(
        "versions",
        help="list dataset versions",
    )
    dataset_versions_parser.add_argument("dataset_id")
    dataset_versions_parser.add_argument("--root", default="datasets")
    dataset_versions_parser.set_defaults(func=_cmd_dataset_versions)

    dataset_materialize_parser = dataset_subparsers.add_parser(
        "materialize-split",
        help="write one split bucket to JSONL",
    )
    dataset_materialize_parser.add_argument("dataset_ref")
    dataset_materialize_parser.add_argument("--split-id", required=True)
    dataset_materialize_parser.add_argument("--split", required=True, choices=["train", "eval", "test"])
    dataset_materialize_parser.add_argument("--output", required=True)
    dataset_materialize_parser.add_argument("--root", default="datasets")
    dataset_materialize_parser.set_defaults(func=_cmd_dataset_materialize_split)

    analyze_parser = subparsers.add_parser("analyze", help="analysis report commands")
    analyze_subparsers = analyze_parser.add_subparsers(
        dest="analyze_command",
        required=True,
    )
    analyze_experiment_parser = analyze_subparsers.add_parser(
        "experiment",
        help="analyze one experiment summary",
    )
    analyze_experiment_parser.add_argument("experiment_id")
    analyze_experiment_parser.add_argument("--experiment-root", default="experiments")
    analyze_experiment_parser.add_argument("--registry-root", default="reports")
    analyze_experiment_parser.set_defaults(func=_cmd_analyze_experiment)

    analyze_benchmark_parser = analyze_subparsers.add_parser(
        "benchmark",
        help="analyze one benchmark metrics file",
    )
    analyze_benchmark_parser.add_argument("benchmark_id")
    analyze_benchmark_parser.add_argument("--benchmark-root", default="benchmarks")
    analyze_benchmark_parser.add_argument("--registry-root", default="reports")
    analyze_benchmark_parser.set_defaults(func=_cmd_analyze_benchmark)

    analyze_compare_parser = analyze_subparsers.add_parser(
        "compare",
        help="compare experiments, benchmarks, and run result files",
    )
    analyze_compare_parser.add_argument("--experiments", nargs="*", default=[])
    analyze_compare_parser.add_argument("--benchmarks", nargs="*", default=[])
    analyze_compare_parser.add_argument("--run-paths", nargs="*", default=[])
    analyze_compare_parser.add_argument("--rank-by")
    analyze_compare_parser.add_argument("--rank-mode", choices=["min", "max"], default="min")
    analyze_compare_parser.add_argument("--experiment-root", default="experiments")
    analyze_compare_parser.add_argument("--benchmark-root", default="benchmarks")
    analyze_compare_parser.add_argument("--registry-root", default="reports")
    analyze_compare_parser.set_defaults(func=_cmd_analyze_compare)

    analyze_list_parser = analyze_subparsers.add_parser(
        "list",
        help="list local analysis reports",
    )
    analyze_list_parser.add_argument("--registry-root", default="reports")
    analyze_list_parser.set_defaults(func=_cmd_analyze_list)

    analyze_show_parser = analyze_subparsers.add_parser(
        "show",
        help="show one analysis record",
    )
    analyze_show_parser.add_argument("analysis_id")
    analyze_show_parser.add_argument("--registry-root", default="reports")
    analyze_show_parser.set_defaults(func=_cmd_analyze_show)

    report_parser = subparsers.add_parser("report", help="report commands")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    report_build_parser = report_subparsers.add_parser(
        "build",
        help="build a Markdown analysis report from config",
    )
    report_build_parser.add_argument("path")
    report_build_parser.add_argument("--registry-root", default="reports")
    report_build_parser.set_defaults(func=_cmd_report_build)

    experiment_parser = subparsers.add_parser("experiment", help="experiment commands")
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_command",
        required=True,
    )
    experiment_run_parser = experiment_subparsers.add_parser(
        "run",
        help="run a local experiment matrix/list",
    )
    experiment_run_parser.add_argument("path")
    experiment_run_parser.add_argument("--registry-root", default="experiments")
    experiment_run_parser.set_defaults(func=_cmd_experiment_run)

    experiment_dry_run_parser = experiment_subparsers.add_parser(
        "dry-run",
        help="expand an experiment without running child jobs",
    )
    experiment_dry_run_parser.add_argument("path")
    experiment_dry_run_parser.set_defaults(func=_cmd_experiment_dry_run)

    experiment_list_parser = experiment_subparsers.add_parser(
        "list",
        help="list local experiments",
    )
    experiment_list_parser.add_argument("--registry-root", default="experiments")
    experiment_list_parser.set_defaults(func=_cmd_experiment_list)

    experiment_show_parser = experiment_subparsers.add_parser(
        "show",
        help="show one experiment record",
    )
    experiment_show_parser.add_argument("experiment_id")
    experiment_show_parser.add_argument("--registry-root", default="experiments")
    experiment_show_parser.set_defaults(func=_cmd_experiment_show)

    sft_parser = subparsers.add_parser("sft", help="SFT commands")
    sft_subparsers = sft_parser.add_subparsers(dest="sft_command", required=True)
    sft_run_parser = sft_subparsers.add_parser("run", help="run SFT from config")
    sft_run_parser.add_argument("path")
    sft_run_parser.set_defaults(func=_cmd_sft_run)
    sft_resume_parser = sft_subparsers.add_parser("resume", help="resume SFT from a full checkpoint")
    sft_resume_parser.add_argument("checkpoint")
    sft_resume_parser.add_argument("--config", dest="config_path")
    sft_resume_parser.set_defaults(func=_cmd_sft_resume)

    pretrain_parser = subparsers.add_parser("pretrain", help="continued pretraining commands")
    pretrain_subparsers = pretrain_parser.add_subparsers(
        dest="pretrain_command",
        required=True,
    )
    pretrain_run_parser = pretrain_subparsers.add_parser(
        "run",
        help="run continued pretraining from config",
    )
    pretrain_run_parser.add_argument("path")
    pretrain_run_parser.set_defaults(func=_cmd_pretrain_run)
    pretrain_resume_parser = pretrain_subparsers.add_parser(
        "resume",
        help="resume continued pretraining from a full checkpoint",
    )
    pretrain_resume_parser.add_argument("checkpoint")
    pretrain_resume_parser.add_argument("--config", dest="config_path")
    pretrain_resume_parser.set_defaults(func=_cmd_pretrain_resume)

    train_parser = subparsers.add_parser("train", help="TinyTrainer commands")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_run_parser = train_subparsers.add_parser("run", help="run TinyTrainer from config")
    train_run_parser.add_argument("path")
    train_run_parser.set_defaults(func=_cmd_train_run)
    train_resume_parser = train_subparsers.add_parser(
        "resume",
        help="resume TinyTrainer from a full checkpoint",
    )
    train_resume_parser.add_argument("checkpoint")
    train_resume_parser.add_argument("--config", dest="config_path")
    train_resume_parser.set_defaults(func=_cmd_train_resume)
    return parser


def _cmd_version(args) -> int:
    print(mopforge.__version__)
    return 0


def _cmd_doctor(args) -> int:
    payload = _doctor_payload(Path(args.root))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"mopforge_version={payload['mopforge_version']}")
        print(f"python_version={payload['python_version']}")
        print(f"torch_available={payload['runtime'].get('torch_available')}")
        print(f"torch_version={payload['runtime'].get('torch_version')}")
        print(f"cuda_available={payload['runtime'].get('cuda_available')}")
        print(f"cuda_device_count={payload['runtime'].get('cuda_device_count')}")
        print(f"mps_available={payload['runtime'].get('mps_available')}")
        for item in payload["writable_dirs"]:
            print(f"writable_dir={item['path']} ok={item['ok']}")
            if item.get("error"):
                print(f"warning={item['error']}")
        for item in payload["configs"]:
            print(f"config={item['path']} present={item['present']}")
        for name, value in payload["optional_dependencies"].items():
            print(f"optional_dependency={name} available={value}")
        if payload["warnings"]:
            for warning in payload["warnings"]:
                print(f"warning={warning}")
        else:
            print("no doctor warnings")
    return 0 if payload["ok"] else 1


def _doctor_payload(root: Path) -> dict:
    runtime = detect_devices()
    writable_dirs = [
        _writable_dir_check(root / name)
        for name in ("runs", "gpu_runs", "artifacts", "outputs")
    ]
    configs = [
        {
            "path": path,
            "present": Path(path).exists(),
        }
        for path in (
            "configs/examples/runtime_cpu.json",
            "configs/examples/trainer_runtime_auto.json",
            "configs/jobs/tiny_gpu_smoke.json",
            "configs/jobs/100m_mop_a100_smoke.json",
        )
    ]
    optional_dependencies = {
        "yaml": _is_module_available("yaml"),
        "tokenizers": _is_module_available("tokenizers"),
        "transformers": _is_module_available("transformers"),
        "torch": bool(runtime.get("torch_available")),
    }
    warnings: list[str] = []
    if not runtime.get("torch_available"):
        warnings.append("PyTorch is not installed; model training commands require optional torch.")
    if not runtime.get("cuda_available"):
        warnings.append("CUDA is not available; GPU jobs will validate/plan or use CPU fallback when allowed.")
    missing_configs = [item["path"] for item in configs if not item["present"]]
    if missing_configs:
        warnings.append("Missing expected config templates: " + ", ".join(missing_configs))
    failed_writes = [item["path"] for item in writable_dirs if not item["ok"]]
    if failed_writes:
        warnings.append("Some output directories are not writable: " + ", ".join(failed_writes))
    return {
        "ok": not failed_writes and not missing_configs,
        "mopforge_version": mopforge.__version__,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "runtime": runtime,
        "writable_dirs": writable_dirs,
        "configs": configs,
        "optional_dependencies": optional_dependencies,
        "warnings": warnings,
    }


def _writable_dir_check(path: Path) -> dict:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".mopforge-doctor-", dir=path, delete=True):
            pass
        return {"path": str(path), "ok": True, "error": None}
    except Exception as exc:
        return {"path": str(path), "ok": False, "error": str(exc)}


def _is_module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _format_cli_exception(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    return str(exc)


def _cmd_modes_list(args) -> int:
    for mode in list_training_modes():
        print(mode)
    return 0


def _cmd_config_write_default(args) -> int:
    config = get_default_config(args.name)
    path = config.save(args.path)
    print(f"wrote {args.name} config to {path}")
    return 0


def _cmd_config_validate(args) -> int:
    config = MoPForgeConfig.load(args.path)
    messages = validate_config_envelope(config)
    errors = [message for message in messages if message.startswith("ERROR:")]
    print(f"kind={config.kind}")
    print("validation=invalid" if errors else "validation=valid")
    if messages:
        for message in messages:
            print(message)
    else:
        print("no warnings or errors")
    return 1 if errors else 0


def _cmd_config_dry_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    summary = dry_run_config(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["runnable_locally"] else 1


def _cmd_runtime_detect(args) -> int:
    payload = detect_devices()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"torch_available={payload.get('torch_available')}")
    print(f"torch_version={payload.get('torch_version')}")
    print(f"cpu_available={payload.get('cpu_available')}")
    print(f"cuda_available={payload.get('cuda_available')}")
    print(f"cuda_version={payload.get('cuda_version')}")
    print(f"cuda_device_count={payload.get('cuda_device_count')}")
    print(f"mps_available={payload.get('mps_available')}")
    for device in payload.get("cuda_devices", []):
        print(
            "cuda_device="
            f"{device.get('index')} name={device.get('name')} "
            f"memory_gb={device.get('total_memory_gb')} "
            f"capability={device.get('capability')}"
        )
    return 0


def _cmd_runtime_dry_run(args) -> int:
    config = RuntimeConfig(
        device=args.device,
        precision=args.precision,
        enable_amp=bool(args.enable_amp),
        allow_tf32=bool(args.allow_tf32),
        deterministic=bool(args.deterministic),
        compile_model=bool(args.compile_model),
        require_device_available=bool(args.require_available),
    )
    try:
        runtime = build_runtime_context(config)
        payload = runtime_metadata(runtime)
        ok = True
    except Exception as exc:
        payload = {
            "requested_device": config.device,
            "requested_precision": config.precision,
            "error": str(exc),
            "warnings": [],
        }
        ok = False
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key in (
            "requested_device",
            "selected_device",
            "requested_precision",
            "selected_precision",
            "amp_enabled",
            "cuda_available",
            "gpu_name",
            "error",
        ):
            if key in payload:
                print(f"{key}={payload.get(key)}")
        for warning in payload.get("warnings", []):
            print(f"warning={warning}")
    return 0 if ok else 1


def _cmd_gpu_validate(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "gpu_train":
        raise ValueError(
            f"gpu validate requires kind='gpu_train', got {envelope.kind!r}. "
            "Use `mopforge config validate` for generic config envelopes."
        )
    try:
        config = gpu_training_config_from_envelope(envelope)
    except Exception as exc:
        print(f"name={envelope.payload.get('name', '<invalid>')}")
        print("validation=invalid")
        print("dry_run=unavailable")
        print("executes_training=False")
        print(f"ERROR: {exc}")
        return 1
    messages = validate_gpu_training_config(config)
    errors = [message for message in messages if message.startswith("ERROR:")]
    print(f"name={config.name}")
    print(f"validation={'invalid' if errors else 'valid'}")
    print("dry_run=available")
    print("executes_training=False")
    print(f"plan_only={bool(config.metadata.get('plan_only'))}")
    if messages:
        for message in messages:
            print(message)
    else:
        print("no warnings or errors")
    return 1 if errors else 0


def _cmd_gpu_estimate(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "gpu_train":
        raise ValueError(
            f"gpu estimate requires kind='gpu_train', got {envelope.kind!r}. "
            "Use a GPU job profile from configs/jobs or `mopforge config write-default gpu_tiny_smoke ...`."
        )
    estimate = estimate_from_config(gpu_training_config_from_envelope(envelope))
    print(json.dumps(estimate.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_gpu_train(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "gpu_train":
        raise ValueError(
            f"gpu train requires kind='gpu_train', got {envelope.kind!r}. "
            "Validate GPU profiles with `mopforge gpu validate <config>` first."
        )
    _ensure_no_validation_errors(envelope)
    config = gpu_training_config_from_envelope(envelope)
    if config.metadata.get("plan_only") and not args.allow_plan_run:
        raise ValueError("This is a planning profile. Re-run with --allow-plan-run to execute it explicitly.")
    updates = {}
    if args.device:
        updates["device"] = args.device
    if args.precision:
        updates["precision"] = args.precision
    if updates:
        payload = config.to_dict()
        payload.update(updates)
        config = type(config).from_dict(payload)
    result = GPUTrainer(config).train()
    selected_device = str(result.runtime_metadata.get("selected_device", ""))
    requested_device = str(result.runtime_metadata.get("requested_device", config.device))
    if selected_device.startswith("cpu") and requested_device in {"auto", "cuda", "mps"}:
        print(
            "WARNING: GPU train selected CPU fallback. "
            "This is useful for smoke tests but does not validate GPU performance."
        )
    print(f"run_id={result.run_id}")
    print(f"status={result.status}")
    print(f"result_path={result.artifacts.get('gpu_training_result_json')}")
    print(f"latest_checkpoint_path={result.artifacts.get('latest_checkpoint_path')}")
    return 0 if result.status == "completed" else 1


def _cmd_gpu_cache_activations(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "gpu_train":
        raise ValueError(
            f"gpu cache-activations requires kind='gpu_train', got {envelope.kind!r}. "
            "Use a GPU sparse training profile as the cache source."
        )
    _ensure_no_validation_errors(envelope)
    config = gpu_training_config_from_envelope(envelope)
    checkpoint_path = _resolve_gpu_checkpoint_arg(args.checkpoint, config.output_root)
    payload = config.to_dict()
    payload.update(
        {
            "resume_from_checkpoint": checkpoint_path,
            "resume_model_only": True,
            "activation_cache_path": None,
            "save_full_checkpoints": False,
        }
    )
    cache_config = type(config).from_dict(payload)
    trainer = GPUTrainer(cache_config)
    trainer.setup()
    result = write_activation_cache(
        model=trainer.model,
        train_loader=trainer.train_loader,
        eval_loader=trainer.eval_loader,
        output_path=args.output,
        runtime=trainer.runtime,
        dtype=args.dtype,
        max_batches=args.max_batches,
        metadata={
            "source_config": str(args.path),
            "source_checkpoint": checkpoint_path,
            "source_checkpoint_sha256": file_sha256(checkpoint_path),
            "config_sha256": config_hash(cache_config),
            "run_id": trainer.run_id,
        },
    )
    print(f"cache_path={result['path']}")
    print(f"cache_format={result['cache_format']}")
    print(f"train_records={result['train_records']}")
    print(f"eval_records={result['eval_records']}")
    print(f"source_checkpoint={checkpoint_path}")
    return 0


def _cmd_gpu_write_warm_sparse_sweep(args) -> int:
    written = write_warm_sparse_sweep_configs(
        output_dir=args.output_dir,
        base_checkpoint=args.base_checkpoint,
        activation_cache_path=args.activation_cache_path,
        dataset_ref=args.dataset_ref,
        dataset_split_id=args.dataset_split_id,
        bottlenecks=list(args.bottlenecks),
        learning_rates=list(args.learning_rates),
        lora_ranks=list(args.lora_ranks),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
    )
    print(f"config_count={len(written)}")
    for path in written:
        print(f"config={path}")
    return 0


def _cmd_gpu_prepare_efficiency_data(args) -> int:
    result = prepare_efficiency_dataset(
        source_path=args.source_path,
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id,
        count_per_category=args.count_per_category,
        verify=args.verify,
        timeout_seconds=args.timeout_seconds,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite,
    )
    print(f"dataset_ref={result['dataset_ref']}")
    print(f"record_count={result['record_count']}")
    print(f"verified_count={result['verified_count']}")
    print(f"split_id={result['split_id']}")
    print(f"split_counts={json.dumps(result['split_counts'], sort_keys=True)}")
    print(f"summary_path={result['summary_path']}")
    return 0


def _cmd_gpu_resume(args) -> int:
    ref = args.checkpoint_or_run_id
    registry = GPURunRegistry()
    checkpoint_path = Path(ref)
    if not checkpoint_path.exists():
        try:
            checkpoint_path = Path(registry.latest_checkpoint(ref))
            config_path = Path(registry.load_record(ref).output_dir) / "config.json"
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not resolve GPU resume reference {ref!r}. "
                "Pass a checkpoint path or run `mopforge gpu list` to find a run ID."
            ) from exc
    else:
        config_path = checkpoint_path.parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find GPU config for resume: {config_path}. "
            "Use a checkpoint from a complete gpu_runs/<run_id>/checkpoints directory."
        )
    config = _gpu_training_config_from_path(config_path)
    payload = dict(config.to_dict())
    payload["resume_from_checkpoint"] = str(checkpoint_path)
    try:
        from mopforge.gpu.checkpointing import load_gpu_checkpoint

        checkpoint = load_gpu_checkpoint(checkpoint_path, map_location="cpu")
        previous_step = int(checkpoint.get("trainer_state", {}).get("global_step", 0))
        payload["max_steps"] = max(int(payload.get("max_steps", 1)), previous_step + 1)
        payload["run_id"] = None
    except Exception:
        pass
    result = GPUTrainer(type(config).from_dict(payload)).train()
    print(f"run_id={result.run_id}")
    print(f"status={result.status}")
    print(f"result_path={result.artifacts.get('gpu_training_result_json')}")
    print(f"resumed_from={checkpoint_path}")
    return 0 if result.status == "completed" else 1


def _cmd_gpu_benchmark(args) -> int:
    try:
        record = GPURunRegistry().load_record(args.run_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find GPU run {args.run_id!r} for benchmark. "
            "Run `mopforge gpu list` first."
        ) from exc
    result_path = Path(record.result_path or Path(record.output_dir) / "gpu_training_result.json")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    output = Path(record.output_dir) / "gpu_benchmark.json"
    payload = {
        "run_id": record.run_id,
        "status": "completed",
        "benchmark_type": "gpu_smoke_summary",
        "runtime": data.get("runtime_metadata", {}),
        "metrics": data.get("metrics", {}),
        "source_result_path": str(result_path),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"run_id={record.run_id}")
    print(f"benchmark_path={output}")
    return 0


def _cmd_gpu_compare_runs(args) -> int:
    from mopforge.gpu.compare import compare_runs, format_table, write_csv, write_json

    rows = compare_runs(list(args.run_ids), gpu_runs_dir=args.gpu_runs_dir)
    json_path = write_json(rows, args.output)
    csv_output = args.output_csv or str(Path(args.output).with_suffix(".csv"))
    csv_path = write_csv(rows, csv_output)
    print(format_table(rows))
    print(f"json_path={json_path}")
    print(f"csv_path={csv_path}")
    return 0


def _cmd_gpu_gate_efficiency(args) -> int:
    report = evaluate_efficiency_gates(
        dense_run=args.dense_run,
        sparse_run=args.sparse_run,
        gpu_runs_dir=args.gpu_runs_dir,
        adapter_baseline_eval_loss=args.adapter_baseline_eval_loss,
        same_quality_eval_delta=args.same_quality_eval_delta,
        generation_pass_delta=args.generation_pass_delta,
        vram_target_gb=args.vram_target_gb,
    )
    output = write_gate_report(report, args.output)
    print(f"overall_passed={report['overall_passed']}")
    if report["failed_required_gates"]:
        print("failed_required_gates=" + ",".join(report["failed_required_gates"]))
    if report["unknown_required_gates"]:
        print("unknown_required_gates=" + ",".join(report["unknown_required_gates"]))
    print(f"gate_report_path={output}")
    return 0 if report["overall_passed"] else 1


def _cmd_gpu_launch_torchrun(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    distributed = DistributedConfig.from_dict(dict(envelope.metadata.get("distributed", {"strategy": "torchrun", "dry_run": True})))
    command = build_torchrun_command(args.path, distributed)
    print("executes=False")
    print("dry_run=True")
    print("command=" + " ".join(command))
    return 0


def _cmd_gpu_list(args) -> int:
    records = GPURunRegistry(args.root).list_runs()
    if not records:
        print("no gpu runs")
        return 0
    for record in records:
        print(f"{record.run_id} status={record.status} name={record.name} result={record.result_path}")
    return 0


def _cmd_gpu_show(args) -> int:
    record = GPURunRegistry(args.root).load_record(args.run_id)
    print(f"run_id={record.run_id}")
    print(f"name={record.name}")
    print(f"status={record.status}")
    print(f"output_dir={record.output_dir}")
    print(f"latest_checkpoint_path={record.latest_checkpoint_path}")
    print(f"metrics_path={record.metrics_path}")
    print(f"result_path={record.result_path}")
    print(f"runtime_path={record.runtime_path}")
    return 0


def _gpu_training_config_from_path(path: str | Path):
    from mopforge.gpu import GPUTrainingConfig

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("kind") == "gpu_train":
        return gpu_training_config_from_envelope(MoPForgeConfig.from_dict(data))
    return GPUTrainingConfig.from_dict(data)


def _cmd_model_register(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "model":
        raise ValueError("model register requires kind='model'.")
    config = model_config_from_envelope(envelope)
    architecture = ModelArchitectureConfig.from_dict(config.architecture)
    manifest = ModelRegistry(args.root).register_model(
        architecture,
        model_id=config.model_id,
        metadata=config.metadata,
    )
    print(f"model_id={manifest.model_id}")
    print(f"version_id={manifest.version_id}")
    print(f"model_type={manifest.architecture.model_type}")
    print(f"manifest_path={manifest.metadata.get('manifest_path')}")
    return 0


def _cmd_model_list(args) -> int:
    records = ModelRegistry(args.root).list_models()
    if not records:
        print("no models")
        return 0
    for record in records:
        print(f"{record.model_id} type={record.model_type} latest={record.latest_version_id} versions={len(record.versions)}")
    return 0


def _cmd_model_show(args) -> int:
    manifest = ModelRegistry(args.root).resolve_model_ref(args.model_ref)
    print(f"model_id={manifest.model_id}")
    print(f"version_id={manifest.version_id}")
    print(f"name={manifest.name}")
    print(f"model_type={manifest.architecture.model_type}")
    print(f"parameter_summary={json.dumps(manifest.parameter_summary, sort_keys=True)}")
    return 0


def _cmd_model_versions(args) -> int:
    for manifest in ModelRegistry(args.root).list_versions(args.model_id):
        print(f"{manifest.version_id} type={manifest.architecture.model_type} params={manifest.parameter_summary.get('total_params')}")
    return 0


def _cmd_model_snapshot(args) -> int:
    manifest = ModelRegistry(args.root).snapshot_model(args.model_id)
    print(f"model_id={manifest.model_id}")
    print(f"version_id={manifest.version_id}")
    return 0


def _cmd_manifest_create(args) -> int:
    envelope = MoPForgeConfig.load(args.config_path)
    spec = ResourceSpec(
        accelerator=args.accelerator,
        num_gpus=args.num_gpus,
        precision=args.precision,
    )
    manifest = plan_run_manifest(envelope, spec, name=args.name, config_ref=args.config_path)
    ManifestRegistry(args.root).create(manifest)
    print(f"manifest_id={manifest.manifest_id}")
    print(f"command={command_text(manifest)}")
    return 0


def _cmd_manifest_dry_run(args) -> int:
    manifest = ManifestRegistry(args.root).load(args.manifest_id)
    print(json.dumps(dry_run_payload(manifest), indent=2, sort_keys=True))
    return 0


def _cmd_manifest_list(args) -> int:
    manifests = ManifestRegistry(args.root).list()
    if not manifests:
        print("no manifests")
        return 0
    for manifest in manifests:
        print(f"{manifest.manifest_id} kind={manifest.run_kind} accelerator={manifest.resource_spec.accelerator} name={manifest.name}")
    return 0


def _cmd_manifest_show(args) -> int:
    manifest = ManifestRegistry(args.root).load(args.manifest_id)
    print(f"manifest_id={manifest.manifest_id}")
    print(f"name={manifest.name}")
    print(f"run_kind={manifest.run_kind}")
    print(f"command={command_text(manifest)}")
    print(f"accelerator={manifest.resource_spec.accelerator}")
    return 0


def _cmd_manifest_export_command(args) -> int:
    path = ManifestRegistry(args.root).export_command(args.manifest_id)
    print(f"command_path={path}")
    return 0


def _cmd_import_results(args) -> int:
    record = import_results(ResultImportConfig(name=args.name, source_path=args.path, output_root=args.root))
    print(f"import_id={record.import_id}")
    print(f"status={record.status}")
    print(f"normalized_results_path={record.normalized_results_path}")
    return 0


def _cmd_import_list(args) -> int:
    records = ResultImportRegistry(args.root).list_imports()
    if not records:
        print("no imports")
        return 0
    for record in records:
        print(f"{record.import_id} status={record.status} rows={record.metadata.get('row_count')} name={record.name}")
    return 0


def _cmd_import_show(args) -> int:
    record = ResultImportRegistry(args.root).load_record(args.import_id)
    print(f"import_id={record.import_id}")
    print(f"name={record.name}")
    print(f"status={record.status}")
    print(f"normalized_results_path={record.normalized_results_path}")
    return 0


def _cmd_ablation_dry_run(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "ablation":
        raise ValueError("ablation dry-run requires kind='ablation'.")
    print(json.dumps(dry_run_ablation(ablation_config_from_envelope(envelope)), indent=2, sort_keys=True))
    return 0


def _cmd_ablation_run(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "ablation":
        raise ValueError("ablation run requires kind='ablation'.")
    result = run_ablation(ablation_config_from_envelope(envelope))
    print(f"ablation_id={result.ablation_id}")
    print(f"status={result.status}")
    print(f"experiment_id={result.experiment_id}")
    print(f"analysis_id={result.analysis_id}")
    print(f"report_path={result.report_path}")
    return 0


def _cmd_ablation_list(args) -> int:
    records = AblationRegistry(args.root).list()
    if not records:
        print("no ablations")
        return 0
    for record in records:
        print(f"{record.ablation_id} status={record.status} experiment={record.experiment_id} name={record.name}")
    return 0


def _cmd_ablation_show(args) -> int:
    record = AblationRegistry(args.root).load(args.ablation_id)
    print(f"ablation_id={record.ablation_id}")
    print(f"name={record.name}")
    print(f"status={record.status}")
    print(f"experiment_id={record.experiment_id}")
    print(f"analysis_id={record.analysis_id}")
    print(f"report_path={record.report_path}")
    return 0


def _cmd_baseline_list(args) -> int:
    for spec in list_baselines():
        print(f"{spec.name} family={spec.family} model_type={spec.model_type}")
    return 0


def _cmd_baseline_show(args) -> int:
    spec = get_baseline(args.name)
    print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_baseline_experiment(args) -> int:
    experiment = build_baseline_experiment_config(list(args.baselines))
    print(json.dumps(experiment.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_stats_summarize(args) -> int:
    rows = json.loads(Path(args.path).read_text(encoding="utf-8"))
    table = make_metric_table(rows, args.group_by, list(args.metric))
    root = Path(args.output_root)
    json_path = write_table_json(table, root / "summary.json")
    csv_path = write_table_csv(table, root / "summary.csv")
    md_path = write_table_markdown(table, root / "summary.md")
    print(f"rows={len(table)}")
    print(f"json_path={json_path}")
    print(f"csv_path={csv_path}")
    print(f"markdown_path={md_path}")
    return 0


def _cmd_paper_build(args) -> int:
    envelope = MoPForgeConfig.load(args.path)
    if envelope.kind != "paper_report":
        raise ValueError("paper build requires kind='paper_report'.")
    record = build_paper_report(paper_report_config_from_envelope(envelope))
    print(f"paper_report_id={record.paper_report_id}")
    print(f"status={record.status}")
    print(f"report_path={record.report_path}")
    return 0


def _cmd_paper_list(args) -> int:
    records = PaperReportRegistry(args.root).list_reports()
    if not records:
        print("no paper reports")
        return 0
    for record in records:
        print(f"{record.paper_report_id} status={record.status} title={record.title}")
    return 0


def _cmd_paper_show(args) -> int:
    record = PaperReportRegistry(args.root).load_record(args.paper_report_id)
    print(f"paper_report_id={record.paper_report_id}")
    print(f"title={record.title}")
    print(f"status={record.status}")
    print(f"report_path={record.report_path}")
    return 0


def _cmd_benchmark_dry_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "benchmark":
        raise ValueError(
            f"benchmark dry-run requires kind='benchmark', got {config.kind!r}."
        )
    summary = dry_run_config(config)
    print(json.dumps(summary["benchmark"], indent=2, sort_keys=True))
    return 0 if summary["runnable_locally"] else 1


def _cmd_benchmark_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "benchmark":
        raise ValueError(f"benchmark run requires kind='benchmark', got {config.kind!r}.")
    _ensure_no_validation_errors(config)
    result = run_benchmark(
        benchmark_config_from_envelope(config),
        registry_root=args.registry_root,
    )
    print(f"benchmark_id={result.benchmark_id}")
    print(f"status={result.status}")
    print(f"benchmark_type={result.benchmark_type}")
    print(f"metrics_path={result.metrics_path}")
    print(f"metrics_csv_path={result.metrics_csv_path}")
    print(f"examples_path={result.examples_path}")
    return 0 if result.status == "completed" else 1


def _cmd_benchmark_list(args) -> int:
    records = BenchmarkRegistry(args.registry_root).list_benchmarks()
    if not records:
        print("no benchmarks")
        return 0
    for record in records:
        print(
            f"{record.benchmark_id} status={record.status} "
            f"type={record.benchmark_type} name={record.name}"
        )
    return 0


def _cmd_benchmark_show(args) -> int:
    record = BenchmarkRegistry(args.registry_root).load_record(args.benchmark_id)
    print(f"benchmark_id={record.benchmark_id}")
    print(f"name={record.name}")
    print(f"benchmark_type={record.benchmark_type}")
    print(f"status={record.status}")
    print(f"metrics_path={record.metrics_path}")
    print(f"metrics_csv_path={record.metrics_csv_path}")
    print(f"examples_path={record.examples_path}")
    return 0


def _cmd_dataset_register(args) -> int:
    manifest = DatasetRegistry(args.root).register_dataset(
        name=args.name,
        kind=args.kind,
        source_paths=list(args.source_paths),
        dataset_id=args.dataset_id,
        description=args.description,
        tags=list(args.tag or []),
        copy_files=bool(args.copy_files),
    )
    _print_dataset_manifest(manifest)
    return 0


def _cmd_dataset_snapshot(args) -> int:
    manifest = DatasetRegistry(args.root).snapshot_dataset(
        args.dataset_id,
        source_paths=list(args.source_path) if args.source_path else None,
        copy_files=bool(args.copy_files),
    )
    _print_dataset_manifest(manifest)
    return 0


def _cmd_dataset_split(args) -> int:
    registry = DatasetRegistry(args.root)
    manifest = registry.load_manifest(args.dataset_id, args.version_id)
    split = create_dataset_split(
        manifest,
        train=args.train,
        eval=args.eval,
        test=args.test,
        seed=args.seed,
        stratify_by=args.stratify_by,
    )
    print(f"dataset_id={split.dataset_id}")
    print(f"version_id={split.version_id}")
    print(f"split_id={split.split_id}")
    print(f"counts={json.dumps(split.counts, sort_keys=True)}")
    return 0


def _cmd_dataset_list(args) -> int:
    records = DatasetRegistry(args.root).list_datasets()
    if not records:
        print("no datasets")
        return 0
    for record in records:
        print(
            f"{record.dataset_id} kind={record.kind} "
            f"latest={record.latest_version_id} versions={len(record.versions)} "
            f"name={record.name}"
        )
    return 0


def _cmd_dataset_show(args) -> int:
    registry = DatasetRegistry(args.root)
    if "@" in args.dataset_ref or Path(args.dataset_ref).exists():
        manifest = registry.resolve_dataset_ref(args.dataset_ref)
        _print_dataset_manifest(manifest)
        return 0
    record = registry.load_dataset_record(args.dataset_ref)
    print(f"dataset_id={record.dataset_id}")
    print(f"name={record.name}")
    print(f"kind={record.kind}")
    print(f"latest_version_id={record.latest_version_id}")
    print(f"versions={','.join(record.versions)}")
    print(f"description={record.description}")
    return 0


def _cmd_dataset_versions(args) -> int:
    manifests = DatasetRegistry(args.root).list_versions(args.dataset_id)
    if not manifests:
        print("no versions")
        return 0
    for manifest in manifests:
        print(
            f"{manifest.version_id} records={manifest.stats.record_count} "
            f"sha256={manifest.combined_sha256} created_at={manifest.created_at}"
        )
    return 0


def _cmd_dataset_materialize_split(args) -> int:
    registry = DatasetRegistry(args.root)
    manifest = registry.resolve_dataset_ref(args.dataset_ref)
    split = load_dataset_split(
        manifest.dataset_id,
        args.split_id,
        version_id=manifest.version_id,
        root=args.root,
    )
    output = write_split_jsonl(manifest, split, args.split, args.output)
    print(f"dataset_id={manifest.dataset_id}")
    print(f"version_id={manifest.version_id}")
    print(f"split_id={split.split_id}")
    print(f"split={args.split}")
    print(f"output={output}")
    return 0


def _cmd_analyze_experiment(args) -> int:
    config = AnalysisConfig(
        name=f"analysis_{args.experiment_id}",
        experiment_ids=[args.experiment_id],
        metrics=["final_eval_loss", "eval_loss_mean", "trainable_ratio"],
        group_by=["mode"],
        rank_by="final_eval_loss",
        rank_mode="min",
        metadata={"experiment_root": args.experiment_root},
    )
    result = run_analysis(config, registry_root=args.registry_root)
    _print_analysis_result(result)
    return 0 if result.status == "completed" else 1


def _cmd_analyze_benchmark(args) -> int:
    config = AnalysisConfig(
        name=f"analysis_{args.benchmark_id}",
        benchmark_ids=[args.benchmark_id],
        metrics=[
            "eval_loss_mean",
            "pass_rate",
            "router_exact_match_rate",
            "trainable_ratio",
        ],
        group_by=["mode"],
        rank_by="eval_loss_mean",
        rank_mode="min",
        metadata={"benchmark_root": args.benchmark_root},
    )
    result = run_analysis(config, registry_root=args.registry_root)
    _print_analysis_result(result)
    return 0 if result.status == "completed" else 1


def _cmd_analyze_compare(args) -> int:
    if not (args.experiments or args.benchmarks or args.run_paths):
        raise ValueError("analyze compare requires at least one source.")
    config = AnalysisConfig(
        name="comparison_analysis",
        experiment_ids=list(args.experiments),
        benchmark_ids=list(args.benchmarks),
        run_paths=list(args.run_paths),
        metrics=[
            "final_eval_loss",
            "eval_loss_mean",
            "pass_rate",
            "router_exact_match_rate",
            "trainable_ratio",
            "trainable_params",
        ],
        group_by=["source_type", "mode"],
        rank_by=args.rank_by,
        rank_mode=args.rank_mode,
        metadata={
            "experiment_root": args.experiment_root,
            "benchmark_root": args.benchmark_root,
        },
    )
    result = run_analysis(config, registry_root=args.registry_root)
    _print_analysis_result(result)
    return 0 if result.status == "completed" else 1


def _cmd_analyze_list(args) -> int:
    records = AnalysisRegistry(args.registry_root).list_analyses()
    if not records:
        print("no analyses")
        return 0
    for record in records:
        print(
            f"{record.analysis_id} status={record.status} "
            f"name={record.name} report_path={record.report_path}"
        )
    return 0


def _cmd_analyze_show(args) -> int:
    record = AnalysisRegistry(args.registry_root).load_record(args.analysis_id)
    print(f"analysis_id={record.analysis_id}")
    print(f"name={record.name}")
    print(f"status={record.status}")
    print(f"report_path={record.report_path}")
    print(f"normalized_results_path={record.normalized_results_path}")
    print(f"comparison_path={record.comparison_path}")
    if record.metadata.get("error"):
        print(f"error={record.metadata['error']}")
    return 0


def _cmd_report_build(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "analysis":
        raise ValueError(f"report build requires kind='analysis', got {config.kind!r}.")
    _ensure_no_validation_errors(config)
    result = run_analysis(analysis_config_from_envelope(config), registry_root=args.registry_root)
    _print_analysis_result(result)
    return 0 if result.status == "completed" else 1


def _cmd_experiment_dry_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "experiment":
        raise ValueError(
            f"experiment dry-run requires kind='experiment', got {config.kind!r}."
        )
    summary = dry_run_config(config)
    print(json.dumps(summary["experiment"], indent=2, sort_keys=True))
    return 0 if summary["runnable_locally"] else 1


def _cmd_experiment_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "experiment":
        raise ValueError(f"experiment run requires kind='experiment', got {config.kind!r}.")
    _ensure_no_validation_errors(config)
    result = run_experiment(
        experiment_config_from_envelope(config),
        registry_root=args.registry_root,
    )
    print(f"experiment_id={result.experiment_id}")
    print(f"status={result.status}")
    print(f"total_runs={result.total_runs}")
    print(f"completed_runs={result.completed_runs}")
    print(f"failed_runs={result.failed_runs}")
    print(f"summary_path={result.summary_path}")
    print(f"summary_csv_path={result.summary_csv_path}")
    return 0


def _cmd_experiment_list(args) -> int:
    records = ExperimentRegistry(args.registry_root).list_experiments()
    if not records:
        print("no experiments")
        return 0
    for record in records:
        print(
            f"{record.experiment_id} status={record.status} "
            f"runs={record.completed_runs}/{record.total_runs} "
            f"failed={record.failed_runs} name={record.name}"
        )
    return 0


def _cmd_experiment_show(args) -> int:
    record = ExperimentRegistry(args.registry_root).load_record(args.experiment_id)
    print(f"experiment_id={record.experiment_id}")
    print(f"name={record.name}")
    print(f"status={record.status}")
    print(f"total_runs={record.total_runs}")
    print(f"completed_runs={record.completed_runs}")
    print(f"failed_runs={record.failed_runs}")
    print(f"run_ids={','.join(record.run_ids)}")
    print(f"summary_path={record.summary_path}")
    if record.metadata.get("summary_csv_path"):
        print(f"summary_csv_path={record.metadata['summary_csv_path']}")
    return 0


def _cmd_sft_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "sft":
        raise ValueError(f"sft run requires kind='sft', got {config.kind!r}.")
    _ensure_no_validation_errors(config)
    result = run_finetune(finetune_config_from_envelope(config))
    result_path = result.artifacts.get("finetune_result_json")
    print(f"run_id={result.run_id}")
    print(f"result_path={result_path}")
    return 0


def _cmd_sft_resume(args) -> int:
    checkpoint_path = _resolve_resume_path(args.checkpoint, args.config_path, "sft")
    payload = load_full_training_checkpoint(checkpoint_path)
    config = _sft_config_for_resume(args.config_path, checkpoint_path, payload)
    result = run_finetune(config)
    result_path = result.artifacts.get("finetune_result_json")
    _print_resume_summary(
        checkpoint_path=checkpoint_path,
        start_step=int(payload.get("global_step", 0)),
        run_id=result.run_id,
        final_step=int(result.metrics.get("global_step", 0)),
        result_path=result_path,
    )
    return 0


def _cmd_pretrain_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "pretrain":
        raise ValueError(f"pretrain run requires kind='pretrain', got {config.kind!r}.")
    _ensure_no_validation_errors(config)
    result = run_continued_pretraining(pretrain_config_from_envelope(config))
    result_path = result.artifacts.get("continued_pretrain_result_json")
    print(f"run_id={result.run_id}")
    print(f"result_path={result_path}")
    return 0


def _cmd_pretrain_resume(args) -> int:
    checkpoint_path = _resolve_resume_path(args.checkpoint, args.config_path, "pretrain")
    payload = load_full_training_checkpoint(checkpoint_path)
    config = _pretrain_config_for_resume(args.config_path, checkpoint_path, payload)
    result = run_continued_pretraining(config)
    result_path = result.artifacts.get("continued_pretrain_result_json")
    _print_resume_summary(
        checkpoint_path=checkpoint_path,
        start_step=int(payload.get("global_step", 0)),
        run_id=result.run_id,
        final_step=int(result.metrics.get("global_step", 0)),
        result_path=result_path,
    )
    return 0


def _cmd_train_run(args) -> int:
    config = MoPForgeConfig.load(args.path)
    if config.kind != "trainer":
        raise ValueError(f"train run requires kind='trainer', got {config.kind!r}.")
    _ensure_no_validation_errors(config)
    result = TinyTrainer(trainer_config_from_envelope(config)).train()
    result_path = result.artifacts.get("trainer_result_json")
    print(f"run_id={result.run_id}")
    print(f"result_path={result_path}")
    return 0


def _cmd_train_resume(args) -> int:
    checkpoint_path = _resolve_resume_path(args.checkpoint, args.config_path, "trainer")
    payload = load_full_training_checkpoint(checkpoint_path)
    config = _trainer_config_for_resume(args.config_path, checkpoint_path, payload)
    result = TinyTrainer(config).train()
    result_path = result.artifacts.get("trainer_result_json")
    _print_resume_summary(
        checkpoint_path=checkpoint_path,
        start_step=int(payload.get("global_step", 0)),
        run_id=result.run_id,
        final_step=int(result.metrics.get("global_step", 0)),
        result_path=result_path,
    )
    return 0


def _ensure_no_validation_errors(config: MoPForgeConfig) -> None:
    errors = [
        message
        for message in validate_config_envelope(config)
        if message.startswith("ERROR:")
    ]
    if errors:
        raise ValueError("; ".join(errors))


def _print_analysis_result(result) -> None:
    print(f"analysis_id={result.analysis_id}")
    print(f"status={result.status}")
    print(f"rows_count={result.rows_count}")
    print(f"report_path={result.report_path}")
    print(f"normalized_results_path={result.normalized_results_path}")
    print(f"comparison_path={result.comparison_path}")
    print(f"record_path={result.record_path}")


def _print_dataset_manifest(manifest) -> None:
    print(f"dataset_id={manifest.dataset_id}")
    print(f"version_id={manifest.version_id}")
    print(f"name={manifest.name}")
    print(f"kind={manifest.kind}")
    print(f"combined_sha256={manifest.combined_sha256}")
    print(f"record_count={manifest.stats.record_count}")
    print(f"manifest_path={manifest.metadata.get('manifest_path')}")


def _resolve_resume_path(
    reference: str,
    config_path: str | None,
    training_kind: str,
) -> Path:
    artifact_root = "artifacts"
    if config_path is not None:
        envelope = MoPForgeConfig.load(config_path)
        if envelope.kind == "trainer":
            artifact_root = trainer_config_from_envelope(envelope).artifact_root
        elif envelope.kind == "sft":
            artifact_root = finetune_config_from_envelope(envelope).artifact_root
        elif envelope.kind == "pretrain":
            artifact_root = pretrain_config_from_envelope(envelope).artifact_root
    return resolve_full_checkpoint_reference(
        reference,
        artifact_root=artifact_root,
        training_kind=training_kind,
    )


def _trainer_config_for_resume(
    config_path: str | None,
    checkpoint_path: Path,
    payload: dict,
) -> TrainerConfig:
    auto_extend = config_path is None
    if config_path is not None:
        envelope = MoPForgeConfig.load(config_path)
        if envelope.kind != "trainer":
            raise ValueError(f"train resume requires kind='trainer', got {envelope.kind!r}.")
        _ensure_no_validation_errors(envelope)
        values = trainer_config_from_envelope(envelope).to_dict()
    else:
        values = dict(payload.get("config") or {})
        if not values:
            raise ValueError("Checkpoint has no TrainerConfig snapshot; pass --config.")
    values["resume_from_checkpoint"] = str(checkpoint_path)
    values["training_kind"] = "trainer"
    _extend_max_steps_for_resume(values, payload, auto_extend=auto_extend)
    return TrainerConfig(**values)


def _sft_config_for_resume(
    config_path: str | None,
    checkpoint_path: Path,
    payload: dict,
) -> FinetuneConfig:
    auto_extend = config_path is None
    if config_path is not None:
        envelope = MoPForgeConfig.load(config_path)
        if envelope.kind != "sft":
            raise ValueError(f"sft resume requires kind='sft', got {envelope.kind!r}.")
        _ensure_no_validation_errors(envelope)
        values = finetune_config_from_envelope(envelope).to_dict()
    else:
        metadata = dict(payload.get("metadata") or {})
        config_snapshot = dict(payload.get("config") or {})
        values = dict(metadata.get("source_config") or config_snapshot.get("source_config") or {})
        if not values:
            raise ValueError("Checkpoint has no FinetuneConfig snapshot; pass --config.")
    values["resume_from_checkpoint"] = str(checkpoint_path)
    _extend_max_steps_for_resume(values, payload, auto_extend=auto_extend)
    return FinetuneConfig(**values)


def _pretrain_config_for_resume(
    config_path: str | None,
    checkpoint_path: Path,
    payload: dict,
) -> ContinuedPretrainConfig:
    auto_extend = config_path is None
    if config_path is not None:
        envelope = MoPForgeConfig.load(config_path)
        if envelope.kind != "pretrain":
            raise ValueError(
                f"pretrain resume requires kind='pretrain', got {envelope.kind!r}."
            )
        _ensure_no_validation_errors(envelope)
        values = pretrain_config_from_envelope(envelope).to_dict()
    else:
        values = dict(payload.get("config") or {})
        if not values:
            raise ValueError(
                "Checkpoint has no ContinuedPretrainConfig snapshot; pass --config."
            )
    values["resume_from_checkpoint"] = str(checkpoint_path)
    _extend_max_steps_for_resume(values, payload, auto_extend=auto_extend)
    return ContinuedPretrainConfig(**values)


def _extend_max_steps_for_resume(
    values: dict,
    payload: dict,
    *,
    auto_extend: bool,
) -> None:
    if not auto_extend:
        return
    start_step = int(payload.get("global_step", 0))
    max_steps = int(values.get("max_steps", 0))
    if max_steps <= start_step:
        values["max_steps"] = start_step + 1


def _print_resume_summary(
    *,
    checkpoint_path: Path,
    start_step: int,
    run_id: str,
    final_step: int,
    result_path: str | None,
) -> None:
    print(f"resumed_from={checkpoint_path}")
    print(f"start_step={start_step}")
    print(f"run_id={run_id}")
    print(f"final_step={final_step}")
    print(f"result_path={result_path}")


def _resolve_gpu_checkpoint_arg(reference: str, output_root: str) -> str:
    candidate = Path(reference)
    if candidate.exists():
        return str(candidate)
    return GPURunRegistry(output_root).latest_checkpoint(reference)


if __name__ == "__main__":
    raise SystemExit(main())
