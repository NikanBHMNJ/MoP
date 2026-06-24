import json

from mopforge.cli.main import main as cli_main
from mopforge.eval import (
    audit_code_contamination,
    evaluate_code_completion,
    load_code_benchmark,
)


def _write_humaneval(path):
    record = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
        "entry_point": "add",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record


def test_humaneval_adapter_and_execution(tmp_path):
    benchmark = tmp_path / "humaneval.jsonl"
    _write_humaneval(benchmark)
    task = load_code_benchmark(benchmark)[0]

    result = evaluate_code_completion(task, "    return a + b\n")

    assert result["syntax_passed"] is True
    assert result["passed"] is True
    assert result["exact_match"] is True


def test_contamination_audit_reports_exact_overlap_and_source_hash(tmp_path):
    benchmark = tmp_path / "humaneval.jsonl"
    record = _write_humaneval(benchmark)
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps({"text": record["prompt"] + record["canonical_solution"]}) + "\n",
        encoding="utf-8",
    )

    report = audit_code_contamination(
        load_code_benchmark(benchmark), [source], ngram_size=5
    )

    assert report["passed"] is False
    assert report["suspected_tasks"] == 1
    assert report["findings"][0]["exact_match"] is True
    assert len(report["source_sha256"][str(source)]) == 64


def test_contamination_cli_writes_machine_readable_report(tmp_path):
    benchmark = tmp_path / "humaneval.jsonl"
    _write_humaneval(benchmark)
    source = tmp_path / "train.txt"
    source.write_text("unrelated arithmetic training document\n", encoding="utf-8")
    output = tmp_path / "audit.json"

    assert (
        cli_main(
            [
                "eval",
                "contamination",
                str(benchmark),
                str(source),
                "--output",
                str(output),
                "--ngram-size",
                "5",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmark_tasks"] == 1
    assert report["suspected_tasks"] == 0
