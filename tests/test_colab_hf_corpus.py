from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.build_colab_hf_corpus import (
    extract_text,
    parse_args,
    rows_to_text_corpus_records,
)


def test_parse_args_defaults_and_overrides() -> None:
    args = parse_args(["--dataset", "demo/data", "--split", "validation", "--max-records", "5"])

    assert args.dataset == "demo/data"
    assert args.split == "validation"
    assert args.text_field == "text"
    assert args.max_records == 5
    assert args.streaming is True


def test_rows_to_text_corpus_records_limits_and_metadata() -> None:
    rows = [{"text": "one"}, {"text": ""}, {"text": "two"}, {"text": "three"}]

    records = rows_to_text_corpus_records(
        rows,
        dataset_name="demo/data",
        split="train",
        text_field="text",
        max_records=2,
    )

    assert [record.text for record in records] == ["one", "two"]
    assert records[0].id == "demo-data-train-00000000"
    assert records[0].source == "demo/data"
    assert records[0].metadata["source_index"] == 0
    assert records[1].metadata["source_index"] == 2


def test_extract_text_supports_nested_field() -> None:
    assert extract_text({"payload": {"text": "hello"}}, "payload.text") == "hello"


def test_helper_cli_works_with_local_jsonl(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "corpus.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "alpha"}),
                json.dumps({"text": "beta"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_colab_hf_corpus.py",
            "--input-jsonl",
            str(input_path),
            "--dataset",
            "local/mock",
            "--split",
            "train",
            "--max-records",
            "2",
            "--output",
            str(output_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert completed.returncode == 0, completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["records_written"] == 2
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "alpha"


def test_colab_notebook_json_is_valid_and_safe_by_default() -> None:
    path = Path("notebooks/train_100m_mopforge_colab.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
    )
    assert "RUN_100M_DENSE = False" in sources
    assert "RUN_100M_MOP = False" in sources
    assert "roneneldan/TinyStories" in sources
    assert "HuggingFaceH4/CodeAlpaca_20K" in sources
    assert "mopforge gpu train configs/jobs/tiny_gpu_smoke.json" in sources
    assert "/content/drive/MyDrive/mopforge_colab_runs/" in sources
