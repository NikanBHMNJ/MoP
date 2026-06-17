"""Tests for local benchmark/evaluation suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mopforge.benchmarks import (
    BenchmarkConfig,
    BenchmarkRecord,
    BenchmarkRegistry,
    count_by_key,
    evaluate_code_correctness,
    evaluate_composite,
    evaluate_loss,
    evaluate_parameter_efficiency,
    evaluate_router,
    flatten_metrics,
    run_benchmark,
    safe_mean,
    safe_rate,
)
from mopforge.cli.main import main as cli_main
from mopforge.configs import (
    MoPForgeConfig,
    benchmark_config_from_envelope,
    dry_run_config,
    validate_config_envelope,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson


def make_lesson(lesson_id: str) -> KnowledgeLesson:
    return KnowledgeLesson(
        id=lesson_id,
        domain="coding",
        skill="debugging",
        subskill="missing-return",
        difficulty=1,
        target_modules=["coding", "debugging"],
        input="def add(a, b):\n    a + b",
        expected_output="def add(a, b):\n    return a + b",
        verification={"type": "python_tests", "status": "verified"},
        metadata={"test_code": "assert add(1, 2) == 3"},
    )


def build_tiny_store(tmp_path) -> None:
    store = IndexedLessonStore(tmp_path / "lessons.jsonl", tmp_path / "lessons.sqlite")
    store.add(make_lesson("lesson-a"))
    store.add(make_lesson("lesson-b"))


def benchmark_config(tmp_path, benchmark_type: str = "loss", **overrides) -> BenchmarkConfig:
    values = {
        "name": f"{benchmark_type}_test",
        "benchmark_type": benchmark_type,
        "lesson_path": str(tmp_path / "lessons.jsonl"),
        "index_path": str(tmp_path / "lessons.sqlite"),
        "output_root": str(tmp_path / "benchmarks"),
        "max_examples": 2,
        "batch_size": 1,
        "max_seq_len": 96,
        "generation_examples": 1,
        "generation_max_new_tokens": 16,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_benchmark_config_validation_and_dict_round_trip(tmp_path) -> None:
    config = benchmark_config(tmp_path)

    loaded = BenchmarkConfig.from_dict(config.to_dict())

    assert loaded == config
    with pytest.raises(ValueError, match="name"):
        BenchmarkConfig(name="")
    with pytest.raises(ValueError, match="benchmark_type"):
        BenchmarkConfig(name="bad", benchmark_type="giant")
    with pytest.raises(ValueError, match="max_examples"):
        BenchmarkConfig(name="bad", max_examples=0)


def test_benchmark_registry_create_save_load_list_and_writes(tmp_path) -> None:
    registry = BenchmarkRegistry(tmp_path / "benchmarks")
    config = benchmark_config(tmp_path)
    record = registry.create_benchmark(config)
    record.status = "completed"
    registry.save_record(record)

    loaded = registry.load_record(record.benchmark_id)
    metrics_path = registry.write_metrics(record.benchmark_id, {"a": 1})
    csv_path = registry.write_metrics_csv(record.benchmark_id, [{"a": 1, "b": {"c": 2}}])
    examples_path = registry.write_examples(record.benchmark_id, [{"lesson_id": "a"}])

    assert loaded.status == "completed"
    assert registry.list_benchmarks()[0].benchmark_id == record.benchmark_id
    assert metrics_path.exists()
    assert csv_path.exists()
    assert examples_path.exists()


def test_benchmark_record_validation() -> None:
    record = BenchmarkRecord(
        benchmark_id="bench-1",
        name="demo",
        benchmark_type="loss",
        status="created",
        created_at="now",
    )

    assert record.to_dict()["benchmark_id"] == "bench-1"
    with pytest.raises(ValueError, match="status"):
        BenchmarkRecord(
            benchmark_id="bench-1",
            name="demo",
            benchmark_type="loss",
            status="odd",
            created_at="now",
        )


def test_metric_helpers() -> None:
    assert safe_mean([1, 2, None]) == 1.5
    assert safe_rate(1, 4) == 0.25
    assert safe_rate(1, 0) == 0.0
    assert count_by_key([{"x": "a"}, {"x": "a"}, {"x": "b"}], "x") == {"a": 2, "b": 1}
    assert flatten_metrics({"a": {"b": 1}, "items": [1, 2]}) == {"a.b": 1, "items": 2}


def test_parameter_efficiency_evaluator_returns_counts(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    metrics = evaluate_parameter_efficiency(
        benchmark_config(
            tmp_path,
            "parameter_efficiency",
            model_type="mop_oracle",
            target_modules=["coding"],
            use_fast_adapters=True,
            metadata={"trainable_policy_mode": "fast_adapters_only"},
        )
    )

    assert metrics["total_params"] > 0
    assert metrics["trainable_params"] > 0
    assert metrics["frozen_params"] > 0
    assert metrics["parameter_groups"]


def test_loss_evaluator_runs_on_tiny_lessons(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    metrics = evaluate_loss(benchmark_config(tmp_path, "loss"))

    assert metrics["benchmark_type"] == "loss"
    assert metrics["eval_loss_count"] >= 1
    assert metrics["examples"] == 2
    assert metrics["finite"] is True


def test_code_correctness_evaluator_returns_metrics_and_examples(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    metrics = evaluate_code_correctness(benchmark_config(tmp_path, "code_correctness"))

    assert metrics["benchmark_type"] == "code_correctness"
    assert metrics["pass_count"] + metrics["fail_count"] == 1
    assert 0.0 <= metrics["pass_rate"] <= 1.0
    assert metrics["examples"]
    assert "generated_preview" in metrics["examples"][0]


def test_router_evaluator_returns_exact_match_fields(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    metrics = evaluate_router(benchmark_config(tmp_path, "router"))

    assert metrics["benchmark_type"] == "router"
    assert "exact_match_count" in metrics
    assert "exact_match_rate" in metrics
    assert metrics["examples"]
    assert metrics["untrained_smoke"] is True


def test_composite_evaluator_returns_nested_metrics(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    metrics = evaluate_composite(
        benchmark_config(
            tmp_path,
            "composite",
            model_type="mop_oracle",
            target_modules=["coding"],
            use_fast_adapters=True,
            metadata={"trainable_policy_mode": "fast_adapters_only"},
        )
    )

    assert "parameter_efficiency" in metrics
    assert "loss" in metrics
    assert "code_correctness" in metrics


def test_run_benchmark_writes_metrics_csv_and_examples(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = run_benchmark(benchmark_config(tmp_path, "code_correctness"))

    assert result.status == "completed"
    assert Path(result.metrics_path).exists()
    assert Path(result.metrics_csv_path).exists()
    assert Path(result.examples_path).exists()
    loaded = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert loaded["benchmark_id"] == result.benchmark_id
    with Path(result.metrics_csv_path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows


def test_run_benchmark_loss_keeps_examples_count_as_metric(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)

    result = run_benchmark(benchmark_config(tmp_path, "loss"))

    assert result.status == "completed"
    assert result.examples_path is None
    assert result.metrics["examples"] == 2


def test_run_benchmark_captures_evaluator_failure(tmp_path) -> None:
    config = benchmark_config(
        tmp_path,
        "loss",
        lesson_path=str(tmp_path / "missing.jsonl"),
    )

    result = run_benchmark(config)

    assert result.status == "failed"
    assert "error" in result.metrics
    assert Path(result.record_path).exists()


def test_benchmark_config_mapping_validation_and_dry_run(tmp_path) -> None:
    envelope = MoPForgeConfig(kind="benchmark", payload=benchmark_config(tmp_path).to_dict())

    mapped = benchmark_config_from_envelope(envelope)
    messages = validate_config_envelope(envelope)
    summary = dry_run_config(envelope)

    assert mapped.benchmark_type == "loss"
    assert not [message for message in messages if message.startswith("ERROR:")]
    assert summary["benchmark"]["benchmark_type"] == "loss"
    assert summary["runnable_locally"] is True


def test_cli_benchmark_dry_run_run_list_show_and_default(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    config_path = MoPForgeConfig(
        kind="benchmark",
        payload=benchmark_config(tmp_path, "parameter_efficiency").to_dict(),
    ).save(tmp_path / "benchmark.json")
    registry_root = tmp_path / "benchmarks"

    assert cli_main(["benchmark", "dry-run", str(config_path)]) == 0
    dry_output = capsys.readouterr().out
    assert '"benchmark_type": "parameter_efficiency"' in dry_output

    assert cli_main([
        "benchmark",
        "run",
        str(config_path),
        "--registry-root",
        str(registry_root),
    ]) == 0
    run_output = capsys.readouterr().out
    assert "benchmark_id=" in run_output
    benchmark_id = [
        line.split("=", 1)[1]
        for line in run_output.splitlines()
        if line.startswith("benchmark_id=")
    ][0]

    assert cli_main(["benchmark", "list", "--registry-root", str(registry_root)]) == 0
    list_output = capsys.readouterr().out
    assert benchmark_id in list_output

    assert cli_main([
        "benchmark",
        "show",
        benchmark_id,
        "--registry-root",
        str(registry_root),
    ]) == 0
    show_output = capsys.readouterr().out
    assert "metrics_path=" in show_output

    default_path = tmp_path / "benchmark_composite.json"
    assert cli_main(["config", "write-default", "benchmark_composite", str(default_path)]) == 0
    assert default_path.exists()


def test_benchmarks_do_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
