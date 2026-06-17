"""Statistical table writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mopforge.statistics.summaries import summarize_by_group


def make_metric_table(rows: list[dict[str, Any]], group_by: str, metrics: list[str]) -> list[dict[str, Any]]:
    return summarize_by_group(rows, group_by, metrics)


def write_table_json(rows: list[dict[str, Any]], path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return str(output)


def write_table_csv(rows: list[dict[str, Any]], path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
    return str(output)


def write_table_markdown(rows: list[dict[str, Any]], path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_table(rows), encoding="utf-8")
    return str(output)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    keys = sorted({key for row in rows for key in row}) or ["value"]
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(key)) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")
