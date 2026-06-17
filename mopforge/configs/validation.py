"""Config envelope validation and runtime mapping helpers."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from mopforge.ablations.config import AblationConfig
from mopforge.analysis.config import AnalysisConfig
from mopforge.benchmarks import BenchmarkConfig
from mopforge.baselines.config import BaselineConfig
from mopforge.configs.io import MoPForgeConfig, SUPPORTED_CONFIG_KINDS
from mopforge.datasets import DatasetConfig
from mopforge.experiments.matrix import ExperimentConfig, expand_experiment_matrix
from mopforge.gpu import GPUTrainingConfig, dry_run_gpu_training_config
from mopforge.importers.result_importer import ResultImportConfig
from mopforge.manifests.run_manifest import ManifestConfig
from mopforge.models.manifest import ModelConfig
from mopforge.papers.config import PaperReportConfig
from mopforge.pretrain import ContinuedPretrainConfig
from mopforge.runtime import RuntimeConfig, build_runtime_context, runtime_metadata
from mopforge.sft import FinetuneConfig, get_training_mode_spec
from mopforge.training import TrainerConfig


def trainer_config_from_envelope(config: MoPForgeConfig) -> TrainerConfig:
    """Build a ``TrainerConfig`` from a trainer envelope."""

    _require_kind(config, "trainer")
    _reject_unknown_fields(config.payload, TrainerConfig)
    return TrainerConfig(**dict(config.payload))


def finetune_config_from_envelope(config: MoPForgeConfig) -> FinetuneConfig:
    """Build a ``FinetuneConfig`` from an SFT envelope."""

    _require_kind(config, "sft")
    _reject_unknown_fields(config.payload, FinetuneConfig)
    return FinetuneConfig(**dict(config.payload))


def pretrain_config_from_envelope(config: MoPForgeConfig) -> ContinuedPretrainConfig:
    """Build a ``ContinuedPretrainConfig`` from a pretrain envelope."""

    _require_kind(config, "pretrain")
    _reject_unknown_fields(config.payload, ContinuedPretrainConfig)
    return ContinuedPretrainConfig(**dict(config.payload))


def experiment_config_from_envelope(config: MoPForgeConfig) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from an experiment envelope."""

    _require_kind(config, "experiment")
    _reject_unknown_fields(config.payload, ExperimentConfig)
    return ExperimentConfig.from_dict(config.payload)


def benchmark_config_from_envelope(config: MoPForgeConfig) -> BenchmarkConfig:
    """Build a ``BenchmarkConfig`` from a benchmark envelope."""

    _require_kind(config, "benchmark")
    _reject_unknown_fields(config.payload, BenchmarkConfig)
    return BenchmarkConfig.from_dict(config.payload)


def analysis_config_from_envelope(config: MoPForgeConfig) -> AnalysisConfig:
    """Build an ``AnalysisConfig`` from an analysis envelope."""

    _require_kind(config, "analysis")
    _reject_unknown_fields(config.payload, AnalysisConfig)
    return AnalysisConfig.from_dict(config.payload)


def dataset_config_from_envelope(config: MoPForgeConfig) -> DatasetConfig:
    """Build a ``DatasetConfig`` from a dataset envelope."""

    _require_kind(config, "dataset")
    _reject_unknown_fields(config.payload, DatasetConfig)
    return DatasetConfig.from_dict(config.payload)


def model_config_from_envelope(config: MoPForgeConfig) -> ModelConfig:
    _require_kind(config, "model")
    _reject_unknown_fields(config.payload, ModelConfig)
    return ModelConfig.from_dict(config.payload)


def manifest_config_from_envelope(config: MoPForgeConfig) -> ManifestConfig:
    _require_kind(config, "manifest")
    _reject_unknown_fields(config.payload, ManifestConfig)
    return ManifestConfig.from_dict(config.payload)


def import_config_from_envelope(config: MoPForgeConfig) -> ResultImportConfig:
    _require_kind(config, "import")
    _reject_unknown_fields(config.payload, ResultImportConfig)
    return ResultImportConfig.from_dict(config.payload)


def ablation_config_from_envelope(config: MoPForgeConfig) -> AblationConfig:
    _require_kind(config, "ablation")
    _reject_unknown_fields(config.payload, AblationConfig)
    return AblationConfig.from_dict(config.payload)


def baseline_config_from_envelope(config: MoPForgeConfig) -> BaselineConfig:
    _require_kind(config, "baseline")
    _reject_unknown_fields(config.payload, BaselineConfig)
    return BaselineConfig.from_dict(config.payload)


def paper_report_config_from_envelope(config: MoPForgeConfig) -> PaperReportConfig:
    _require_kind(config, "paper_report")
    _reject_unknown_fields(config.payload, PaperReportConfig)
    return PaperReportConfig.from_dict(config.payload)


def runtime_config_from_envelope(config: MoPForgeConfig) -> RuntimeConfig:
    _require_kind(config, "runtime")
    _reject_unknown_fields(config.payload, RuntimeConfig)
    return RuntimeConfig.from_dict(config.payload)


def gpu_training_config_from_envelope(config: MoPForgeConfig) -> GPUTrainingConfig:
    _require_kind(config, "gpu_train")
    _reject_unknown_fields(config.payload, GPUTrainingConfig)
    return GPUTrainingConfig.from_dict(config.payload)


def validate_config_envelope(config: MoPForgeConfig) -> list[str]:
    """Return validation messages for a config envelope.

    Messages are prefixed with ``ERROR:`` or ``WARNING:``. Unknown payload fields
    are errors; this keeps config files inspectable and typo-resistant.
    """

    messages: list[str] = []
    if config.kind not in SUPPORTED_CONFIG_KINDS:
        messages.append(f"ERROR: unknown kind {config.kind!r}.")
        return messages
    if config.kind in {"queue"}:
        messages.append(f"ERROR: kind {config.kind!r} is reserved for later.")
        return messages

    payload = config.payload
    _check_positive_int(payload, "max_steps", messages)
    _check_positive_int(payload, "batch_size", messages)
    _check_positive_int(payload, "max_seq_len", messages)
    _check_optional_positive_int(payload, "checkpoint_every_steps", messages)
    _check_positive_int(payload, "max_examples", messages)
    _check_positive_int(payload, "generation_examples", messages)
    _check_positive_int(payload, "generation_max_new_tokens", messages)
    _check_string_field(payload, "lesson_path", messages)
    _check_string_field(payload, "index_path", messages)
    _check_string_field(payload, "corpus_path", messages)
    _check_string_field(payload, "run_registry_root", messages)
    _check_string_field(payload, "artifact_root", messages)
    _check_tokenizer_spec_path(payload, messages)
    _check_resume_checkpoint_path(payload, messages)
    _check_benchmark_checkpoint_path(payload, messages)
    _check_dataset_source_paths(payload, messages)
    _check_device(payload, messages)

    try:
        if config.kind == "trainer":
            _reject_unknown_fields(payload, TrainerConfig)
            trainer_config_from_envelope(config)
        elif config.kind == "sft":
            _validate_sft_payload(payload, messages)
            _reject_unknown_fields(payload, FinetuneConfig)
            finetune_config_from_envelope(config)
        elif config.kind == "pretrain":
            _reject_unknown_fields(payload, ContinuedPretrainConfig)
            pretrain_config_from_envelope(config)
        elif config.kind == "experiment":
            _reject_unknown_fields(payload, ExperimentConfig)
            experiment_config = experiment_config_from_envelope(config)
            expanded = expand_experiment_matrix(experiment_config)
            if not expanded:
                messages.append("ERROR: experiment expands to zero runs.")
            if len(expanded) > 16:
                messages.append(
                    f"WARNING: experiment expands to {len(expanded)} local runs."
                )
            for index, child in enumerate(expanded):
                child_messages = validate_config_envelope(child)
                for message in child_messages:
                    if message.startswith("ERROR:"):
                        messages.append(f"ERROR: child run {index}: {message[7:]}")
        elif config.kind == "benchmark":
            _reject_unknown_fields(payload, BenchmarkConfig)
            benchmark_config_from_envelope(config)
        elif config.kind == "analysis":
            _reject_unknown_fields(payload, AnalysisConfig)
            analysis_config = analysis_config_from_envelope(config)
            if not (
                analysis_config.experiment_ids
                or analysis_config.benchmark_ids
                or analysis_config.run_paths
                or analysis_config.metadata.get("allow_empty_sources")
            ):
                messages.append(
                    "ERROR: analysis requires at least one experiment_id, benchmark_id, or run_path."
                )
        elif config.kind == "dataset":
            _reject_unknown_fields(payload, DatasetConfig)
            dataset_config = dataset_config_from_envelope(config)
            if dataset_config.action == "register" and not dataset_config.source_paths:
                messages.append("ERROR: dataset register requires source_paths.")
            if dataset_config.action == "snapshot" and not (
                dataset_config.dataset_id or dataset_config.dataset_ref
            ):
                messages.append("ERROR: dataset snapshot requires dataset_id or dataset_ref.")
            if dataset_config.action == "split" and not (
                dataset_config.dataset_ref or dataset_config.dataset_id
            ):
                messages.append("ERROR: dataset split requires dataset_ref or dataset_id.")
            if abs(
                dataset_config.split_train
                + dataset_config.split_eval
                + dataset_config.split_test
                - 1.0
            ) > 1e-6:
                messages.append("ERROR: dataset split ratios must sum to 1.0.")
        elif config.kind == "model":
            _reject_unknown_fields(payload, ModelConfig)
            model_config_from_envelope(config)
        elif config.kind == "manifest":
            _reject_unknown_fields(payload, ManifestConfig)
            manifest_config_from_envelope(config)
        elif config.kind == "import":
            _reject_unknown_fields(payload, ResultImportConfig)
            import_config_from_envelope(config)
        elif config.kind == "ablation":
            _reject_unknown_fields(payload, AblationConfig)
            ablation_config_from_envelope(config)
        elif config.kind == "baseline":
            _reject_unknown_fields(payload, BaselineConfig)
            baseline_config_from_envelope(config)
        elif config.kind == "stats":
            if not isinstance(payload.get("metrics", []), list):
                messages.append("ERROR: stats metrics must be a list.")
            _check_string_field(payload, "input_path", messages)
            _check_string_field(payload, "group_by", messages)
            _check_string_field(payload, "output_root", messages)
        elif config.kind == "paper_report":
            _reject_unknown_fields(payload, PaperReportConfig)
            paper_report_config_from_envelope(config)
        elif config.kind == "runtime":
            _reject_unknown_fields(payload, RuntimeConfig)
            runtime_config_from_envelope(config)
        elif config.kind == "gpu_train":
            _reject_unknown_fields(payload, GPUTrainingConfig)
            gpu_training_config_from_envelope(config)
    except Exception as exc:
        messages.append(f"ERROR: {exc}")
    return messages


def dry_run_config(config: MoPForgeConfig) -> dict[str, Any]:
    """Return a small resolved-runtime summary without running training."""

    warnings = validate_config_envelope(config)
    runtime_config: dict[str, Any] | None = None
    try:
        if config.kind == "trainer":
            runtime_config = trainer_config_from_envelope(config).to_dict()
        elif config.kind == "sft":
            runtime_config = finetune_config_from_envelope(config).to_dict()
        elif config.kind == "pretrain":
            runtime_config = pretrain_config_from_envelope(config).to_dict()
        elif config.kind == "experiment":
            runtime_config = experiment_config_from_envelope(config).to_dict()
        elif config.kind == "benchmark":
            runtime_config = benchmark_config_from_envelope(config).to_dict()
        elif config.kind == "analysis":
            runtime_config = analysis_config_from_envelope(config).to_dict()
        elif config.kind == "dataset":
            runtime_config = dataset_config_from_envelope(config).to_dict()
        elif config.kind == "model":
            runtime_config = model_config_from_envelope(config).to_dict()
        elif config.kind == "manifest":
            runtime_config = manifest_config_from_envelope(config).to_dict()
        elif config.kind == "import":
            runtime_config = import_config_from_envelope(config).to_dict()
        elif config.kind == "ablation":
            runtime_config = ablation_config_from_envelope(config).to_dict()
        elif config.kind == "baseline":
            runtime_config = baseline_config_from_envelope(config).to_dict()
        elif config.kind == "stats":
            runtime_config = dict(config.payload)
        elif config.kind == "paper_report":
            runtime_config = paper_report_config_from_envelope(config).to_dict()
        elif config.kind == "runtime":
            runtime_config = runtime_config_from_envelope(config).to_dict()
        elif config.kind == "gpu_train":
            runtime_config = gpu_training_config_from_envelope(config).to_dict()
    except Exception:
        runtime_config = None

    error_count = sum(1 for message in warnings if message.startswith("ERROR:"))
    expected_roots = {
        "run_registry_root": _payload_or_runtime(
            config.payload,
            runtime_config,
            "run_registry_root",
            "runs",
        ),
        "artifact_root": _payload_or_runtime(
            config.payload,
            runtime_config,
            "artifact_root",
            "artifacts",
        ),
    }
    runtime_summary = _runtime_dry_run_summary(config.kind, runtime_config or config.payload)
    runtime_error = bool(runtime_summary.get("error"))
    selected_device = runtime_summary.get("selected_device")
    device = selected_device or _payload_or_runtime(config.payload, runtime_config, "device", "cpu")
    experiment_summary = None
    if config.kind == "experiment" and runtime_config is not None:
        try:
            expanded = expand_experiment_matrix(ExperimentConfig.from_dict(runtime_config))
            experiment_summary = {
                "expanded_run_count": len(expanded),
                "first_runs": [run.to_dict() for run in expanded[:3]],
                "supported_child_kinds": sorted({run.kind for run in expanded}),
            }
        except Exception as exc:
            experiment_summary = {"error": str(exc)}
    if config.kind == "experiment":
        device = "cpu"
    benchmark_summary = None
    if config.kind == "benchmark" and runtime_config is not None:
        benchmark_summary = {
            "benchmark_type": runtime_config.get("benchmark_type"),
            "max_examples": runtime_config.get("max_examples"),
            "generation_examples": runtime_config.get("generation_examples"),
            "expected_output_root": runtime_config.get("output_root", "benchmarks"),
            "checkpoint_path": runtime_config.get("checkpoint_path"),
            "checkpoint_provided": bool(runtime_config.get("checkpoint_path")),
            "source_run_id": runtime_config.get("run_id"),
        }
    analysis_summary = None
    if config.kind == "analysis" and runtime_config is not None:
        analysis_summary = {
            "experiment_count": len(runtime_config.get("experiment_ids", [])),
            "benchmark_count": len(runtime_config.get("benchmark_ids", [])),
            "run_path_count": len(runtime_config.get("run_paths", [])),
            "output_root": runtime_config.get("output_root", "reports"),
            "rank_by": runtime_config.get("rank_by"),
            "rank_mode": runtime_config.get("rank_mode", "min"),
            "metrics": list(runtime_config.get("metrics", [])),
            "group_by": list(runtime_config.get("group_by", [])),
            "source_note": dict(runtime_config.get("metadata", {})).get("source_note"),
            "locally_runnable": True,
        }
        if not (
            runtime_config.get("experiment_ids")
            or runtime_config.get("benchmark_ids")
            or runtime_config.get("run_paths")
        ):
            analysis_summary["warning"] = (
                "No sources configured; report build will produce a zero-row scaffold."
            )
        device = "cpu"
    dataset_summary = None
    if config.kind == "dataset" and runtime_config is not None:
        dataset_summary = {
            "action": runtime_config.get("action"),
            "dataset_id": runtime_config.get("dataset_id"),
            "dataset_ref": runtime_config.get("dataset_ref"),
            "kind": runtime_config.get("kind"),
            "source_count": len(runtime_config.get("source_paths", [])),
            "output_root": runtime_config.get("output_root", "datasets"),
            "copy_files": bool(runtime_config.get("copy_files")),
            "split_ratios": {
                "train": runtime_config.get("split_train"),
                "eval": runtime_config.get("split_eval"),
                "test": runtime_config.get("split_test"),
            },
            "split_seed": runtime_config.get("split_seed"),
            "stratify_by": runtime_config.get("stratify_by"),
            "locally_runnable": True,
        }
        device = "cpu"
    research_summary = None
    if config.kind in {"model", "manifest", "import", "ablation", "baseline", "stats", "paper_report", "runtime"} and runtime_config is not None:
        research_summary = {
            "kind": config.kind,
            "output_root": runtime_config.get("output_root"),
            "action": runtime_config.get("action"),
            "name": runtime_config.get("name") or runtime_config.get("title"),
            "locally_runnable": True,
            "gpu_execution": False,
        }
        if config.kind != "runtime":
            device = "cpu"
    gpu_summary = None
    if config.kind == "gpu_train" and runtime_config is not None:
        gpu_config = GPUTrainingConfig.from_dict(runtime_config)
        gpu_summary = dry_run_gpu_training_config(gpu_config)
        expected_roots = {
            "run_registry_root": gpu_config.output_root,
            "artifact_root": gpu_config.artifact_root,
        }
        device = runtime_summary.get("selected_device") or gpu_config.device
    checkpointing = {
        "save_checkpoints": _payload_or_runtime(
            config.payload,
            runtime_config,
            "save_checkpoints",
            True,
        ),
        "save_full_checkpoints": _payload_or_runtime(
            config.payload,
            runtime_config,
            "save_full_checkpoints",
            True,
        ),
        "checkpoint_every_steps": _payload_or_runtime(
            config.payload,
            runtime_config,
            "checkpoint_every_steps",
            None,
        ),
        "resume_from_checkpoint": _payload_or_runtime(
            config.payload,
            runtime_config,
            "resume_from_checkpoint",
            None,
        ),
    }
    return {
        "kind": config.kind,
        "version": config.version,
        "runtime_config": runtime_config,
        "warnings": warnings,
        "expected_output_roots": expected_roots,
        "checkpointing": checkpointing,
        "experiment": experiment_summary,
        "benchmark": benchmark_summary,
        "analysis": analysis_summary,
        "dataset": dataset_summary,
        "research": research_summary,
        "runtime": runtime_summary,
        "gpu": gpu_summary,
        "runnable_locally": error_count == 0 and not runtime_error and str(device).startswith("cpu"),
    }


def _require_kind(config: MoPForgeConfig, expected: str) -> None:
    if config.kind != expected:
        raise ValueError(f"Config kind must be {expected!r}, got {config.kind!r}.")


def _reject_unknown_fields(payload: dict[str, Any], config_cls) -> None:
    allowed = {field.name for field in fields(config_cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown fields for {config_cls.__name__}: {', '.join(unknown)}."
        )


def _validate_sft_payload(payload: dict[str, Any], messages: list[str]) -> None:
    mode = str(payload.get("mode", "sft_full"))
    try:
        get_training_mode_spec(mode)
    except ValueError as exc:
        messages.append(f"ERROR: {exc}")
        return
    target_modules = payload.get("target_modules")
    if mode in {"sft_module", "sft_generated"} and not target_modules:
        messages.append(f"ERROR: {mode} requires target_modules.")
    if mode == "sft_adapter" and payload.get("use_fast_adapters") is False:
        messages.append("ERROR: sft_adapter must enable use_fast_adapters.")
    if mode == "sft_generated" and payload.get("use_generated_params") is False:
        messages.append("ERROR: sft_generated must enable use_generated_params.")


def _check_positive_int(payload: dict[str, Any], field_name: str, messages: list[str]) -> None:
    if field_name not in payload:
        return
    value = payload[field_name]
    if type(value) is not int or value <= 0:
        messages.append(f"ERROR: {field_name} must be a positive integer.")


def _check_optional_positive_int(
    payload: dict[str, Any],
    field_name: str,
    messages: list[str],
) -> None:
    if field_name not in payload or payload[field_name] is None:
        return
    _check_positive_int(payload, field_name, messages)


def _check_string_field(payload: dict[str, Any], field_name: str, messages: list[str]) -> None:
    if field_name not in payload or payload[field_name] is None:
        return
    if not isinstance(payload[field_name], str):
        messages.append(f"ERROR: {field_name} must be a string.")


def _check_tokenizer_spec_path(payload: dict[str, Any], messages: list[str]) -> None:
    path = payload.get("tokenizer_spec_path")
    if path is None:
        return
    if not isinstance(path, str):
        messages.append("ERROR: tokenizer_spec_path must be a string.")
        return
    if not Path(path).exists():
        messages.append(f"ERROR: tokenizer_spec_path does not exist: {path}")


def _check_resume_checkpoint_path(payload: dict[str, Any], messages: list[str]) -> None:
    path = payload.get("resume_from_checkpoint")
    if path is None:
        return
    if not isinstance(path, str):
        messages.append("ERROR: resume_from_checkpoint must be a string.")
        return
    if not path.strip():
        messages.append("ERROR: resume_from_checkpoint must be a non-empty string.")
        return
    candidate = Path(path)
    looks_like_path = bool(candidate.suffix) or "\\" in path or "/" in path
    if looks_like_path and not candidate.exists():
        messages.append(f"WARNING: resume checkpoint path does not exist: {path}")


def _check_benchmark_checkpoint_path(payload: dict[str, Any], messages: list[str]) -> None:
    path = payload.get("checkpoint_path")
    if path is None:
        return
    if not isinstance(path, str):
        messages.append("ERROR: checkpoint_path must be a string.")
        return
    if not path.strip():
        messages.append("ERROR: checkpoint_path must be a non-empty string.")
        return
    if not Path(path).exists():
        messages.append(f"ERROR: checkpoint_path does not exist: {path}")


def _check_dataset_source_paths(payload: dict[str, Any], messages: list[str]) -> None:
    source_paths = payload.get("source_paths")
    if source_paths is None:
        return
    if not isinstance(source_paths, list):
        messages.append("ERROR: source_paths must be a list of strings.")
        return
    for path in source_paths:
        if not isinstance(path, str) or not path.strip():
            messages.append("ERROR: source_paths must contain non-empty strings.")
            continue
        if not Path(path).exists():
            messages.append(f"WARNING: dataset source path does not exist yet: {path}")


def _check_device(payload: dict[str, Any], messages: list[str]) -> None:
    device = payload.get("device")
    precision = payload.get("precision", "fp32")
    if device is None and "precision" not in payload:
        return
    try:
        RuntimeConfig(
            device=device or "cpu",
            precision=precision,
            enable_amp=bool(payload.get("enable_amp", payload.get("use_amp", False))),
            allow_tf32=bool(payload.get("allow_tf32", False)),
            deterministic=bool(payload.get("deterministic", False)),
            compile_model=bool(payload.get("compile_model", False)),
            require_device_available=bool(payload.get("require_device_available", True)),
        )
    except Exception as exc:
        messages.append(f"ERROR: runtime config: {exc}")


def _runtime_dry_run_summary(kind: str, values: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(values or {})
    if kind not in {"trainer", "sft", "pretrain", "benchmark", "runtime", "gpu_train"}:
        return {}
    config = RuntimeConfig(
        device=values.get("device", "cpu"),
        precision=values.get("precision", "fp32"),
        enable_amp=bool(values.get("enable_amp", values.get("use_amp", False))),
        allow_tf32=bool(values.get("allow_tf32", False)),
        deterministic=bool(values.get("deterministic", False)),
        compile_model=bool(values.get("compile_model", False)),
        require_device_available=bool(values.get("require_device_available", True)),
    )
    try:
        runtime = build_runtime_context(config)
        return runtime_metadata(runtime)
    except Exception as exc:
        return {
            "requested_device": config.device,
            "selected_device": None,
            "requested_precision": config.precision,
            "selected_precision": None,
            "amp_enabled": False,
            "warnings": [],
            "error": str(exc),
        }


def _payload_or_runtime(
    payload: dict[str, Any],
    runtime_config: dict[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    if runtime_config is not None and key in runtime_config:
        return runtime_config[key]
    return payload.get(key, default)
