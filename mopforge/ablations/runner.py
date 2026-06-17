"""Ablation runner built on experiment and analysis layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopforge.ablations.config import AblationConfig
from mopforge.ablations.registry import AblationRegistry, _write_json
from mopforge.analysis import AnalysisConfig, run_analysis
from mopforge.configs.io import MoPForgeConfig
from mopforge.experiments.matrix import ExperimentConfig
from mopforge.experiments.runner import run_experiment


@dataclass(slots=True)
class AblationResult:
    ablation_id: str
    status: str
    experiment_id: str | None
    analysis_id: str | None
    report_path: str
    summary_path: str


def expand_ablation_variants(config: AblationConfig) -> list[MoPForgeConfig]:
    base = dict(config.base_config)
    if "kind" not in base:
        raise ValueError("base_config must be a MoPForgeConfig envelope dictionary.")
    runs = []
    for variant in config.variants:
        envelope = MoPForgeConfig.from_dict(_deepcopy(base))
        payload = dict(envelope.payload)
        for key, value in variant.overrides.items():
            _set_dotted(payload, key, value)
        envelope.payload = payload
        envelope.metadata.update({"ablation_variant": variant.name, "variant_tags": list(variant.tags)})
        runs.append(envelope)
    return runs


def dry_run_ablation(config: AblationConfig) -> dict[str, Any]:
    runs = expand_ablation_variants(config)
    return {"name": config.name, "variant_count": len(runs), "variants": [run.to_dict() for run in runs]}


def run_ablation(config: AblationConfig) -> AblationResult:
    config = AblationConfig.from_dict(config.to_dict())
    registry = AblationRegistry(config.output_root)
    record = registry.create(config.name)
    directory = registry.ablation_dir(record.ablation_id)
    runs = expand_ablation_variants(config)
    _write_json(directory / "ablation.json", config.to_dict())
    _write_json(directory / "variants.json", [run.to_dict() for run in runs])
    experiment = run_experiment(
        ExperimentConfig(
            name=config.name,
            kind="list",
            description=config.description,
            runs=[run.to_dict() for run in runs],
            max_runs=len(runs),
            metadata={"ablation_id": record.ablation_id},
        )
    )
    (directory / "experiment_id.txt").write_text(experiment.experiment_id, encoding="utf-8")
    analysis = run_analysis(
        AnalysisConfig(
            name=f"{config.name}_analysis",
            experiment_ids=[experiment.experiment_id],
            rank_by=config.rank_by,
            rank_mode=config.rank_mode,
            group_by=["mode", "model_type"],
        )
    )
    (directory / "analysis_id.txt").write_text(analysis.analysis_id, encoding="utf-8")
    summary = {
        "ablation_id": record.ablation_id,
        "status": "completed" if experiment.failed_runs == 0 else "completed_with_failures",
        "experiment_id": experiment.experiment_id,
        "analysis_id": analysis.analysis_id,
        "variant_count": len(runs),
        "completed_runs": experiment.completed_runs,
        "failed_runs": experiment.failed_runs,
        "analysis_report_path": analysis.report_path,
    }
    summary_path = _write_json(directory / "summary.json", summary)
    report_path = directory / "report.md"
    report_path.write_text(
        f"# {config.name}\n\nAblation status: {summary['status']}\n\n"
        f"Experiment: `{experiment.experiment_id}`\n\n"
        f"Analysis: `{analysis.analysis_id}`\n\n"
        "CPU smoke ablation only. Metrics are not model-quality claims.\n",
        encoding="utf-8",
    )
    record.status = summary["status"]
    record.experiment_id = experiment.experiment_id
    record.analysis_id = analysis.analysis_id
    record.report_path = str(report_path)
    record.metadata["summary_path"] = str(summary_path)
    registry.save(record)
    return AblationResult(record.ablation_id, record.status, experiment.experiment_id, analysis.analysis_id, str(report_path), str(summary_path))


def _set_dotted(payload: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = payload
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot apply override through non-dict path: {key}")
        current = child
    current[parts[-1]] = value


def _deepcopy(value):
    return json.loads(json.dumps(value))
