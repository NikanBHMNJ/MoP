"""Tests for paper-style reports."""

from __future__ import annotations

from pathlib import Path

from mopforge.cli.main import main as cli_main
from mopforge.configs import get_default_config
from mopforge.papers import PaperReportConfig, PaperReportRegistry, build_paper_report


def test_paper_report_build_sections(tmp_path) -> None:
    record = build_paper_report(PaperReportConfig(title="Demo", output_root=str(tmp_path / "papers"), dataset_refs=["data"], model_refs=["model"]))
    text = Path(record.report_path).read_text(encoding="utf-8")
    assert "## Abstract" in text
    assert "## Reproducibility Checklist" in text
    assert PaperReportRegistry(tmp_path / "papers").load_record(record.paper_report_id).title == "Demo"


def test_paper_cli(tmp_path, capsys) -> None:
    config = get_default_config("paper_report_smoke")
    config.payload["output_root"] = str(tmp_path / "papers")
    path = config.save(tmp_path / "paper.json")
    assert cli_main(["paper", "build", str(path)]) == 0
    output = capsys.readouterr().out
    report_id = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("paper_report_id=")][0]
    assert cli_main(["paper", "list", "--root", str(tmp_path / "papers")]) == 0
    assert report_id in capsys.readouterr().out
    assert cli_main(["paper", "show", report_id, "--root", str(tmp_path / "papers")]) == 0
