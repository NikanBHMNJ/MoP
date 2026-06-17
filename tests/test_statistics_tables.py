"""Tests for statistics helpers and CLI."""

from __future__ import annotations

import json

from mopforge.cli.main import main as cli_main
from mopforge.statistics import make_metric_table, mean, median, percent_change, stderr, stddev


def test_statistics_helpers() -> None:
    assert mean([1, 2, None]) == 1.5
    assert median([1, 3, 2]) == 2
    assert stddev([1, 1]) == 0
    assert stderr([1, 1]) == 0
    assert percent_change(2, 1) == 100


def test_metric_table_and_cli(tmp_path, capsys) -> None:
    rows = [{"mode": "a", "loss": 1.0}, {"mode": "a", "loss": 2.0}, {"mode": "b", "loss": 3.0}]
    assert make_metric_table(rows, "mode", ["loss"])[0]["loss_mean"] == 1.5
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert cli_main(["stats", "summarize", str(path), "--group-by", "mode", "--metric", "loss", "--output-root", str(tmp_path / "stats")]) == 0
    assert "json_path=" in capsys.readouterr().out
