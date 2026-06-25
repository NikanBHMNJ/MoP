from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.build_colab_hf_corpus import (
    extract_text,
    parse_args,
    resolve_hf_split,
    rows_to_text_corpus_records,
)


def test_parse_args_defaults_and_overrides() -> None:
    args = parse_args(["--dataset", "demo/data", "--split", "validation", "--max-records", "5"])

    assert args.dataset == "demo/data"
    assert args.split == "validation"
    assert args.text_field == "text"
    assert args.max_records == 5
    assert args.streaming is False
    assert args.output == "data/colab_hf_corpus.jsonl"
    assert args.domain == "code"


def test_resolve_hf_split_bounds_non_streaming_rows() -> None:
    assert resolve_hf_split("train", streaming=False, max_records=6000) == "train[:6000]"
    assert resolve_hf_split("train", streaming=True, max_records=6000) == "train"
    assert resolve_hf_split("train[:100]", streaming=False, max_records=6000) == "train[:100]"


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
    assert summary["dataset"] == "input"
    assert summary["records_written"] == 2
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "alpha"


def test_current_notebooks_json_is_valid_and_current() -> None:
    notebook_paths = [
        Path("notebooks/colab_l4_verified_code_repair_100m.ipynb"),
        Path("notebooks/colab_a100_1b_feasibility_probe.ipynb"),
        Path("notebooks/colab_h100_2b_readiness.ipynb"),
    ]

    sources = []
    for path in notebook_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        sources.append(
            "\n".join(
                "".join(cell.get("source", []))
                for cell in payload.get("cells", [])
            )
        )
    combined_source = "\n".join(sources)

    assert "roneneldan/" + "Tiny" + "Stories" not in combined_source
    assert "train_100m_mopforge_colab" not in combined_source
    legacy_suffix = "".join(["g", "o", "a", "l"])
    assert f"colab_l4_{legacy_suffix}" not in combined_source
    assert f"colab_a100_{legacy_suffix}" not in combined_source
    assert f"colab_h100_{legacy_suffix}" not in combined_source
    assert "mopforge gpu train" in combined_source
    assert "'gpu', 'probe'" in combined_source or '"gpu", "probe"' in combined_source
    assert "'tokenizer', 'train-bpe'" in combined_source or '"tokenizer", "train-bpe"' in combined_source
    assert "files.download" in combined_source


def test_l4_notebook_requires_explicit_run_flags() -> None:
    path = Path("notebooks/colab_l4_verified_code_repair_100m.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
    )

    assert "RUN_DENSE = True" in sources
    assert "RUN_MOP_FULL = True" in sources
    assert "RUN_CACHED_ADAPTER_128 = True" in sources
    assert "RUN_CACHED_LORA_8 = True" in sources
    assert "RUN_CACHED_LORA_16 = False" in sources
