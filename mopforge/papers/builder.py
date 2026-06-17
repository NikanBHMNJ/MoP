"""Build conservative paper-style Markdown report scaffolds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mopforge.papers.config import PaperReportConfig


@dataclass(slots=True)
class PaperReportRecord:
    paper_report_id: str
    title: str
    status: str
    report_path: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "paper_report_id": self.paper_report_id,
            "title": self.title,
            "status": self.status,
            "report_path": self.report_path,
            "created_at": self.created_at,
        }


class PaperReportRegistry:
    def __init__(self, root: str | Path = "paper_reports") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_reports(self) -> list[PaperReportRecord]:
        records = []
        for path in self.root.iterdir() if self.root.exists() else []:
            record_path = path / "record.json"
            if record_path.exists():
                data = json.loads(record_path.read_text(encoding="utf-8"))
                records.append(PaperReportRecord(**data))
        return sorted(records, key=lambda record: (record.created_at, record.paper_report_id))

    def load_record(self, paper_report_id: str) -> PaperReportRecord:
        path = self.root / paper_report_id / "record.json"
        if not path.exists():
            raise FileNotFoundError(f"Paper report does not exist: {paper_report_id}")
        return PaperReportRecord(**json.loads(path.read_text(encoding="utf-8")))


def build_paper_report(config: PaperReportConfig) -> PaperReportRecord:
    config = PaperReportConfig.from_dict(config.to_dict())
    report_id = _id(config.title)
    directory = Path(config.output_root) / report_id
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    (directory / "paper_report.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = directory / "report.md"
    report_path.write_text(_markdown(config, report_id), encoding="utf-8")
    record = PaperReportRecord(
        paper_report_id=report_id,
        title=config.title,
        status="completed",
        report_path=str(report_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (directory / "record.json").write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return record


def _markdown(config: PaperReportConfig, report_id: str) -> str:
    sections = [
        f"# {config.title}",
        "",
        config.subtitle,
        "",
        f"Report ID: `{report_id}`",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Abstract",
        "",
        config.abstract or "CPU-smoke research scaffold. Metrics are not model-quality claims.",
        "",
        "## Motivation",
        "",
        "MoP-Forge explores stable shared cores, routed parameter groups, adapters, generated parameters, and controlled local research workflows.",
        "",
        "## Method / MoP Framing",
        "",
        "This scaffold documents local artifacts for future Mixture-of-Parameters research without claiming production-scale results.",
        "",
        "## Data Provenance",
        "",
        _bullets(config.dataset_refs),
        "",
        "## Model Configurations",
        "",
        _bullets(config.model_refs),
        "",
        "## Experiment Setup",
        "",
        _bullets(config.experiment_ids + config.manifest_ids),
        "",
        "## Benchmark Protocols",
        "",
        _bullets(config.benchmark_ids),
        "",
        "## Results",
        "",
        _bullets(config.analysis_ids),
        "",
        "## Statistical Summaries",
        "",
        "Simple statistical tables may be attached from analysis/statistics artifacts when available.",
        "",
        "## Analysis Notes",
        "",
        "Interpret results conservatively. Tiny CPU smoke metrics are plumbing checks, not quality claims.",
        "",
    ]
    if config.include_limitations:
        sections.extend(
            [
                "## Limitations",
                "",
                "- No GPU training was executed.",
                "- No statistical significance claims are made.",
                "- No PDF or LaTeX output is generated.",
                "- Source metrics are only as meaningful as their local artifacts.",
                "",
            ]
        )
    if config.include_reproducibility_checklist:
        sections.extend(
            [
                "## Reproducibility Checklist",
                "",
                "- Dataset refs listed.",
                "- Model refs listed.",
                "- Experiment/benchmark/analysis IDs listed.",
                "- Local commands and manifests should be archived with results.",
                "",
            ]
        )
    sections.extend(["## Artifact Index", "", _artifact_index(config), ""])
    return "\n".join(line for line in sections if line is not None)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- `{item}`" for item in items) if items else "- none"


def _artifact_index(config: PaperReportConfig) -> str:
    items = {
        "analysis": config.analysis_ids,
        "experiments": config.experiment_ids,
        "benchmarks": config.benchmark_ids,
        "datasets": config.dataset_refs,
        "models": config.model_refs,
        "manifests": config.manifest_ids,
    }
    return "\n".join(f"- {key}: {', '.join(values) if values else 'none'}" for key, values in items.items())


def _id(title: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in title).strip("-")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe or 'paper'}-{uuid4().hex[:8]}"
