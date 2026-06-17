"""Paper-style Markdown report scaffolds."""

from mopforge.papers.builder import (
    PaperReportRecord,
    PaperReportRegistry,
    build_paper_report,
)
from mopforge.papers.config import PaperReportConfig

__all__ = [
    "PaperReportConfig",
    "PaperReportRecord",
    "PaperReportRegistry",
    "build_paper_report",
]
