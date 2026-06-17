"""Plan portable research run manifests without execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mopforge.configs.io import MoPForgeConfig
from mopforge.configs.validation import dry_run_config
from mopforge.manifests.resources import ResourceSpec
from mopforge.manifests.run_manifest import ResearchRunManifest


COMMANDS = {
    "trainer": ["mopforge", "train", "run"],
    "sft": ["mopforge", "sft", "run"],
    "pretrain": ["mopforge", "pretrain", "run"],
    "benchmark": ["mopforge", "benchmark", "run"],
    "experiment": ["mopforge", "experiment", "run"],
    "analysis": ["mopforge", "report", "build"],
}
RUN_KIND = {
    "trainer": "train",
    "sft": "sft",
    "pretrain": "pretrain",
    "benchmark": "benchmark",
    "experiment": "experiment",
    "analysis": "analysis",
}


def plan_run_manifest(
    config: MoPForgeConfig,
    resources: ResourceSpec,
    name=None,
    config_ref: str | None = None,
) -> ResearchRunManifest:
    """Create a no-execution run manifest from a config envelope."""

    if config.kind not in COMMANDS:
        raise ValueError(f"Unsupported manifest config kind: {config.kind}")
    resources = ResourceSpec.from_dict(resources.to_dict())
    payload = dict(config.payload)
    config_ref = config_ref or str(payload.get("config_ref") or "")
    command = list(COMMANDS[config.kind])
    command.append(config_ref or "<config-path>")
    dry = dry_run_config(config)
    roots = dry.get("expected_output_roots", {})
    runtime = dict(dry.get("runtime") or {})
    warnings = _runtime_resource_warnings(resources, runtime)
    manifest_id = _manifest_id(name or payload.get("run_name") or config.kind)
    return ResearchRunManifest(
        manifest_id=manifest_id,
        name=str(name or payload.get("run_name") or f"{config.kind}_manifest"),
        created_at=datetime.now(timezone.utc).isoformat(),
        run_kind=RUN_KIND[config.kind],
        config_ref=config_ref or None,
        config_payload=config.to_dict(),
        model_ref=payload.get("model_ref"),
        dataset_ref=payload.get("dataset_ref") or payload.get("corpus_dataset_ref"),
        benchmark_refs=list(payload.get("benchmark_refs", [])) if isinstance(payload.get("benchmark_refs", []), list) else [],
        resource_spec=resources,
        command=command,
        expected_outputs=[str(value) for value in roots.values() if value],
        metadata={"dry_run": dry, "runtime": runtime, "warnings": warnings, "gpu_execution": False},
    )


def command_text(manifest: ResearchRunManifest) -> str:
    return " ".join(_quote(part) for part in manifest.command)


def dry_run_payload(manifest: ResearchRunManifest) -> dict:
    return {
        "manifest_id": manifest.manifest_id,
        "name": manifest.name,
        "command": manifest.command,
        "command_text": command_text(manifest),
        "resource_spec": manifest.resource_spec.to_dict(),
        "executes_gpu": False,
        "planned_gpu": manifest.resource_spec.accelerator not in {"none", "cpu"},
        "runtime": dict(manifest.metadata.get("runtime", {})),
        "warnings": list(manifest.metadata.get("warnings", [])),
        "expected_outputs": list(manifest.expected_outputs),
    }


def config_from_path_or_payload(config_ref: str | None, payload: dict) -> MoPForgeConfig:
    if config_ref:
        return MoPForgeConfig.load(config_ref)
    return MoPForgeConfig.from_dict(payload)


def _manifest_id(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(name)).strip("-")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe or 'manifest'}-{uuid4().hex[:8]}"


def _quote(part: str) -> str:
    return json.dumps(part) if any(ch.isspace() for ch in part) else part


def _runtime_resource_warnings(resources: ResourceSpec, runtime: dict) -> list[str]:
    warnings: list[str] = []
    requested_device = str(runtime.get("requested_device") or "")
    if resources.accelerator not in {"none", "cpu"} and requested_device in {"", "cpu"}:
        warnings.append("Manifest resource requests a GPU accelerator but config runtime requests CPU.")
    if requested_device.startswith("cuda") and resources.accelerator in {"none", "cpu"}:
        warnings.append("Config runtime requests CUDA but manifest resources are CPU/none.")
    for key in ("requested_device", "requested_precision", "amp_enabled", "allow_tf32"):
        if key in runtime:
            resources.metadata.setdefault(f"runtime_{key}", runtime[key])
    return warnings
