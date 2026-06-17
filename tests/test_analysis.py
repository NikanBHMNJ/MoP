"""Tests for local result analysis and report generation."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.analysis import (
    AnalysisConfig,
    AnalysisRecord,
    AnalysisRegistry,
    build_markdown_report,
    compare_results,
    filter_rows,
    group_rows,
    load_benchmark_metrics,
    load_experiment_summary,
    load_run_result,
    markdown_table,
    normalize_benchmark_metrics,
    normalize_experiment_rows,
    normalize_run_result,
    numeric_delta,
    rank_rows,
    run_analysis,
    summarize_group,
)
from mopforge.cli.main import main as cli_main
from mopforge.configs import (
    MoPForgeConfig,
    analysis_config_from_envelope,
    dry_run_config,
    validate_config_envelope,
)


def experiment_rows() -> list[dict]:
    return [
        {
            "experiment_id": "exp-a",
            "index": 0,
            "kind": "sft",
            "status": "completed",
            "run_id": "run-adapter",
            "mode": "sft_adapter",
            "model_type": "mop_oracle",
            "trainable_policy_mode": "fast_adapters_only",
            "final_train_loss": "0.7",
            "final_eval_loss": "0.4",
            "finite": "true",
            "result_path": "runs/run-adapter/finetune_result.json",
            "error": None,
        },
        {
            "experiment_id": "exp-a",
            "index": 1,
            "kind": "sft",
            "status": "completed",
            "run_id": "run-generated",
            "mode": "sft_generated",
            "model_type": "mop_oracle",
            "trainable_policy_mode": "generated_params_only",
            "final_train_loss": 0.6,
            "final_eval_loss": 0.3,
            "finite": True,
            "result_path": "runs/run-generated/finetune_result.json",
            "error": None,
        },
    ]


def benchmark_metrics() -> dict:
    return {
        "benchmark_id": "bench-a",
        "benchmark_name": "composite",
        "benchmark_type": "composite",
        "status": "completed",
        "source_run_id": "run-generated",
        "parameter_efficiency": {
            "benchmark_type": "parameter_efficiency",
            "model_type": "mop_oracle",
            "total_params": 100,
            "trainable_params": 10,
            "frozen_params": 90,
            "trainable_ratio": 0.1,
            "use_fast_adapters": True,
            "use_generated_params": False,
        },
        "loss": {
            "benchmark_type": "loss",
            "model_type": "mop_oracle",
            "eval_loss_mean": 0.25,
            "finite": True,
        },
        "code_correctness": {
            "benchmark_type": "code_correctness",
            "pass_count": 1,
            "fail_count": 1,
            "pass_rate": 0.5,
        },
        "router": {
            "benchmark_type": "router",
            "exact_match_rate": 0.75,
        },
    }


def write_experiment(root: Path, experiment_id: str = "exp-a") -> Path:
    path = root / experiment_id
    path.mkdir(parents=True)
    summary = {"experiment_id": experiment_id, "rows": experiment_rows()}
    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def write_benchmark(root: Path, benchmark_id: str = "bench-a") -> Path:
    path = root / benchmark_id
    path.mkdir(parents=True)
    metrics = benchmark_metrics()
    metrics["benchmark_id"] = benchmark_id
    metrics_path = path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return metrics_path


def write_run_result(tmp_path: Path, name: str = "finetune_result.json") -> Path:
    path = tmp_path / name
    payload = {
        "run_id": "run-a",
        "mode": "sft_adapter",
        "artifacts": {"finetune_result_json": str(path)},
        "metrics": {
            "finetune_mode": "sft_adapter",
            "finetune_config": {
                "model_type": "mop_oracle",
                "use_fast_adapters": True,
                "use_generated_params": False,
            },
            "eval_loss_mean": 0.2,
            "train_loss_last": 0.5,
            "finite": True,
            "parameter_counts": {"total": 50, "trainable": 5, "frozen": 45},
            "trainable_policy": {"mode": "fast_adapters_only"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def analysis_config(tmp_path: Path, **overrides) -> AnalysisConfig:
    values = {
        "name": "analysis_test",
        "experiment_ids": ["exp-a"],
        "benchmark_ids": ["bench-a"],
        "output_root": str(tmp_path / "reports"),
        "metrics": ["final_eval_loss", "eval_loss_mean", "pass_rate", "trainable_ratio"],
        "group_by": ["source_type"],
        "rank_by": "final_eval_loss",
        "metadata": {
            "experiment_root": str(tmp_path / "experiments"),
            "benchmark_root": str(tmp_path / "benchmarks"),
        },
    }
    values.update(overrides)
    return AnalysisConfig(**values)


def test_analysis_config_validation_and_dict_round_trip(tmp_path) -> None:
    config = analysis_config(tmp_path)

    loaded = AnalysisConfig.from_dict(config.to_dict())

    assert loaded == config
    try:
        AnalysisConfig(name="")
        assert False
    except ValueError as exc:
        assert "name" in str(exc)
    try:
        AnalysisConfig(name="bad", rank_mode="middle")
        assert False
    except ValueError as exc:
        assert "rank_mode" in str(exc)


def test_analysis_registry_create_save_load_list(tmp_path) -> None:
    registry = AnalysisRegistry(tmp_path / "reports")
    record = registry.create_analysis(analysis_config(tmp_path))
    record.status = "completed"
    registry.save_record(record)

    loaded = registry.load_record(record.analysis_id)

    assert loaded.status == "completed"
    assert registry.list_analyses()[0].analysis_id == record.analysis_id


def test_analysis_record_validation() -> None:
    record = AnalysisRecord(
        analysis_id="analysis-a",
        name="demo",
        status="created",
        created_at="now",
    )

    assert record.to_dict()["analysis_id"] == "analysis-a"
    try:
        AnalysisRecord(analysis_id="x", name="demo", status="odd", created_at="now")
        assert False
    except ValueError as exc:
        assert "status" in str(exc)


def test_load_experiment_summary_by_path_and_id(tmp_path) -> None:
    summary_path = write_experiment(tmp_path / "experiments")

    by_path = load_experiment_summary(summary_path)
    by_id = load_experiment_summary("exp-a", root=tmp_path / "experiments")

    assert by_path[0]["run_id"] == "run-adapter"
    assert by_id[1]["mode"] == "sft_generated"


def test_load_benchmark_metrics_by_path_and_id(tmp_path) -> None:
    metrics_path = write_benchmark(tmp_path / "benchmarks")

    by_path = load_benchmark_metrics(metrics_path)
    by_id = load_benchmark_metrics("bench-a", root=tmp_path / "benchmarks")

    assert by_path["benchmark_type"] == "composite"
    assert by_id["loss"]["eval_loss_mean"] == 0.25


def test_normalize_experiment_rows() -> None:
    rows = normalize_experiment_rows(experiment_rows(), source_id="exp-a")

    assert rows[0]["source_type"] == "experiment"
    assert rows[0]["final_eval_loss"] == 0.4
    assert rows[0]["finite"] is True


def test_normalize_benchmark_metrics() -> None:
    rows = normalize_benchmark_metrics(benchmark_metrics(), source_id="bench-a")

    assert rows[0]["source_type"] == "benchmark"
    assert rows[0]["eval_loss_mean"] == 0.25
    assert rows[0]["pass_rate"] == 0.5
    assert rows[0]["router_exact_match_rate"] == 0.75
    assert any(row["mode"] == "parameter_efficiency" for row in rows)


def test_normalize_run_result_files_for_shapes(tmp_path) -> None:
    finetune = load_run_result(write_run_result(tmp_path, "finetune_result.json"))
    trainer_path = write_run_result(tmp_path, "trainer_result.json")
    pretrain_path = write_run_result(tmp_path, "continued_pretrain_result.json")

    assert normalize_run_result(finetune)["kind"] == "sft"
    assert normalize_run_result(load_run_result(trainer_path), str(trainer_path))["kind"] in {"sft", "trainer"}
    assert normalize_run_result(load_run_result(pretrain_path), str(pretrain_path))["kind"] == "pretrain"


def test_ranking_min_max_filtering_and_delta() -> None:
    rows = normalize_experiment_rows(experiment_rows())

    assert rank_rows(rows, "final_eval_loss", "min")[0]["mode"] == "sft_generated"
    assert rank_rows(rows, "final_eval_loss", "max")[0]["mode"] == "sft_adapter"
    assert filter_rows(rows, {"mode": "sft_adapter"})[0]["run_id"] == "run-adapter"
    assert numeric_delta(0.3, 0.4) == -0.10000000000000003


def test_grouping_summaries_and_compare_deltas() -> None:
    rows = normalize_experiment_rows(experiment_rows())
    comparison = compare_results(
        rows,
        metrics=["final_eval_loss"],
        group_by=["mode"],
        rank_by="final_eval_loss",
        baseline_filter={"mode": "sft_adapter"},
    )

    assert "mode=sft_adapter" in group_rows(rows, ["mode"])
    assert summarize_group(rows, ["final_eval_loss"])["final_eval_loss_min"] == 0.3
    assert comparison["best_row"]["mode"] == "sft_generated"
    assert comparison["deltas_vs_baseline"][1]["final_eval_loss_delta"] < 0


def test_markdown_table_and_report_sections(tmp_path) -> None:
    rows = normalize_experiment_rows(experiment_rows())
    config = analysis_config(tmp_path)
    comparison = compare_results(rows, rank_by="final_eval_loss")

    table = markdown_table(rows, ["mode", "final_eval_loss"])
    report = build_markdown_report(config, rows, comparison)

    assert "| mode | final_eval_loss |" in table
    assert "## Sources" in report
    assert "## Ranking" in report
    assert "## Group Summaries" in report
    assert "## Limitations" in report


def test_run_analysis_writes_artifacts(tmp_path) -> None:
    write_experiment(tmp_path / "experiments")
    write_benchmark(tmp_path / "benchmarks")

    result = run_analysis(analysis_config(tmp_path))

    report_dir = Path(result.record_path).parent
    assert result.status == "completed"
    assert Path(result.normalized_results_path).exists()
    assert (report_dir / "normalized_results.csv").exists()
    assert Path(result.comparison_path).exists()
    assert (report_dir / "comparison.csv").exists()
    assert Path(result.report_path).exists()


def test_failed_analysis_writes_failed_record(tmp_path) -> None:
    result = run_analysis(
        analysis_config(
            tmp_path,
            experiment_ids=["missing"],
            benchmark_ids=[],
            metadata={"experiment_root": str(tmp_path / "experiments")},
        )
    )

    record = AnalysisRegistry(tmp_path / "reports").load_record(result.analysis_id)
    assert result.status == "failed"
    assert record.metadata["error"]


def test_config_envelope_mapping_validation_and_dry_run(tmp_path) -> None:
    envelope = MoPForgeConfig(kind="analysis", payload=analysis_config(tmp_path).to_dict())

    mapped = analysis_config_from_envelope(envelope)
    messages = validate_config_envelope(envelope)
    summary = dry_run_config(envelope)

    assert mapped.name == "analysis_test"
    assert not [message for message in messages if message.startswith("ERROR:")]
    assert summary["analysis"]["experiment_count"] == 1
    assert summary["runnable_locally"] is True


def test_cli_analyze_experiment_benchmark_compare_list_show_and_report_build(tmp_path, capsys) -> None:
    write_experiment(tmp_path / "experiments")
    write_benchmark(tmp_path / "benchmarks")
    report_root = tmp_path / "reports"

    assert cli_main([
        "analyze",
        "experiment",
        "exp-a",
        "--experiment-root",
        str(tmp_path / "experiments"),
        "--registry-root",
        str(report_root),
    ]) == 0
    experiment_output = capsys.readouterr().out
    assert "analysis_id=" in experiment_output

    assert cli_main([
        "analyze",
        "benchmark",
        "bench-a",
        "--benchmark-root",
        str(tmp_path / "benchmarks"),
        "--registry-root",
        str(report_root),
    ]) == 0
    benchmark_output = capsys.readouterr().out
    assert "report_path=" in benchmark_output

    assert cli_main([
        "analyze",
        "compare",
        "--experiments",
        "exp-a",
        "--benchmarks",
        "bench-a",
        "--experiment-root",
        str(tmp_path / "experiments"),
        "--benchmark-root",
        str(tmp_path / "benchmarks"),
        "--registry-root",
        str(report_root),
        "--rank-by",
        "final_eval_loss",
    ]) == 0
    compare_output = capsys.readouterr().out
    analysis_id = [
        line.split("=", 1)[1]
        for line in compare_output.splitlines()
        if line.startswith("analysis_id=")
    ][0]

    assert cli_main(["analyze", "list", "--registry-root", str(report_root)]) == 0
    assert analysis_id in capsys.readouterr().out
    assert cli_main([
        "analyze",
        "show",
        analysis_id,
        "--registry-root",
        str(report_root),
    ]) == 0
    assert "comparison_path=" in capsys.readouterr().out

    run_path = write_run_result(tmp_path)
    config_path = MoPForgeConfig(
        kind="analysis",
        payload=AnalysisConfig(
            name="report_build_test",
            run_paths=[str(run_path)],
            output_root=str(report_root),
            rank_by="final_eval_loss",
        ).to_dict(),
    ).save(tmp_path / "analysis.json")
    assert cli_main(["report", "build", str(config_path), "--registry-root", str(report_root)]) == 0
    assert "report_path=" in capsys.readouterr().out


def test_cli_config_write_default_analysis_composite_report(tmp_path, capsys) -> None:
    path = tmp_path / "analysis_composite_report.json"

    assert cli_main(["config", "write-default", "analysis_composite_report", str(path)]) == 0
    assert path.exists()
    assert cli_main(["config", "dry-run", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"analysis"' in output
    assert "zero-row scaffold" in output


def test_analysis_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
