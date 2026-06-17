"""Artifact detection for local result imports."""

from __future__ import annotations

from pathlib import Path


KNOWN_ARTIFACT_NAMES = {
    "trainer_result.json": "run",
    "finetune_result.json": "run",
    "continued_pretrain_result.json": "run",
    "metrics.json": "metrics",
    "summary.json": "experiment",
    "summary.csv": "experiment",
    "report.md": "analysis_report",
}


def detect_artifacts(source_path: str | Path) -> list[dict]:
    """Detect known result artifact files under a file or directory."""

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Import source does not exist: {source_path}")
    candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    artifacts = []
    for item in sorted(candidates):
        kind = KNOWN_ARTIFACT_NAMES.get(item.name)
        if kind is None:
            continue
        if item.name == "metrics.json" and item.parent.name.startswith("202"):
            kind = "benchmark_or_run_metrics"
        artifacts.append({"path": str(item), "artifact_type": kind, "name": item.name})
    return artifacts
