"""Paper-style report config schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mopforge.configs.io import MoPForgeConfig


@dataclass(slots=True)
class PaperReportConfig:
    title: str
    subtitle: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    analysis_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    benchmark_ids: list[str] = field(default_factory=list)
    dataset_refs: list[str] = field(default_factory=list)
    model_refs: list[str] = field(default_factory=list)
    manifest_ids: list[str] = field(default_factory=list)
    output_root: str = "paper_reports"
    include_limitations: bool = True
    include_reproducibility_checklist: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be non-empty.")
        for field_name in ("authors", "analysis_ids", "experiment_ids", "benchmark_ids", "dataset_refs", "model_refs", "manifest_ids"):
            value = getattr(self, field_name)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field_name} must be a list of strings.")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be non-empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperReportConfig":
        return cls(**dict(data))

    def save(self, path) -> object:
        return MoPForgeConfig(kind="paper_report", payload=self.to_dict()).save(path)
