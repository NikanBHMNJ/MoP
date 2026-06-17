"""Build a MoP-Forge TextCorpusRecord JSONL file from HF or local JSONL rows.

The Colab path uses ``datasets.load_dataset(..., streaming=True)`` by default.
Tests and offline verification can use ``--input-jsonl`` to exercise the same
conversion code without internet access or Hugging Face dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mopforge.pretrain import TextCorpusRecord


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face dataset or local JSONL rows to MoP-Forge corpus JSONL."
    )
    parser.add_argument("--dataset", default="roneneldan/TinyStories", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train", help="dataset split")
    parser.add_argument("--text-field", default="text", help="row field containing text")
    parser.add_argument("--max-records", type=int, default=1000, help="maximum records to write")
    parser.add_argument("--output", default="data/colab_tinystories_corpus.jsonl", help="output JSONL path")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="use HF streaming")
    parser.add_argument("--input-jsonl", help="optional local JSONL input for offline conversion")
    parser.add_argument("--id-prefix", help="record id prefix; defaults to a slug of dataset/split")
    parser.add_argument("--source", help="record source label; defaults to dataset name")
    parser.add_argument("--domain", default="story", help="optional domain metadata")
    parser.add_argument("--language", default="en", help="optional language metadata")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_records <= 0:
        raise ValueError("--max-records must be positive.")
    rows = (
        read_jsonl_rows(args.input_jsonl)
        if args.input_jsonl
        else load_hf_rows(args.dataset, args.split, streaming=bool(args.streaming))
    )
    records = rows_to_text_corpus_records(
        rows,
        dataset_name=args.dataset,
        split=args.split,
        text_field=args.text_field,
        max_records=args.max_records,
        id_prefix=args.id_prefix,
        source=args.source,
        domain=args.domain,
        language=args.language,
    )
    output = write_records(records, args.output)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "text_field": args.text_field,
        "streaming": bool(args.streaming) and not args.input_jsonl,
        "input_jsonl": args.input_jsonl,
        "output": str(output),
        "records_written": len(records),
        "preview_ids": [record.id for record in records[:3]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def load_hf_rows(dataset_name: str, split: str, *, streaming: bool = True) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for remote dataset loading. "
            "Install it in Colab with `pip install datasets`, or use --input-jsonl."
        ) from exc
    return load_dataset(dataset_name, split=split, streaming=streaming)


def read_jsonl_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {input_path}:{line_number}.")
            yield row


def rows_to_text_corpus_records(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    text_field: str,
    max_records: int,
    id_prefix: str | None = None,
    source: str | None = None,
    domain: str | None = "story",
    language: str | None = "en",
) -> list[TextCorpusRecord]:
    if max_records <= 0:
        raise ValueError("max_records must be positive.")
    prefix = id_prefix or f"{slugify(dataset_name)}-{slugify(split)}"
    source_label = source or dataset_name
    records: list[TextCorpusRecord] = []
    for index, row in enumerate(rows):
        if len(records) >= max_records:
            break
        text = extract_text(row, text_field)
        if not text.strip():
            continue
        records.append(
            TextCorpusRecord(
                id=f"{prefix}-{len(records):08d}",
                text=text,
                source=source_label,
                domain=domain,
                language=language,
                metadata={
                    "dataset": dataset_name,
                    "split": split,
                    "source_index": index,
                    "text_field": text_field,
                    "row_keys": sorted(str(key) for key in row.keys()),
                },
            )
        )
    return records


def extract_text(row: dict[str, Any], text_field: str) -> str:
    value: Any = row
    for part in text_field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"text field {text_field!r} not found in row keys: {sorted(row.keys())}")
        value = value[part]
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def write_records(records: list[TextCorpusRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return output


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "dataset"


if __name__ == "__main__":
    raise SystemExit(main())
