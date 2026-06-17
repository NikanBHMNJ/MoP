"""Tests for the tiny experiment harness."""

from __future__ import annotations

import json
import math

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.experiments import (
    TinyExperimentConfig,
    run_tiny_comparison,
    split_lessons,
    write_results,
)
from mopforge.experiments.tiny_compare import REQUIRED_RESULT_KEYS


def test_tiny_experiment_config_defaults_are_cpu_safe() -> None:
    config = TinyExperimentConfig()

    assert config.batch_size == 2
    assert config.train_steps == 3
    assert config.router_train_steps == 3
    assert config.eval_batches == 2
    assert config.max_seq_len == 512
    assert config.run_generation_eval is False
    assert config.generation_eval_examples == 3
    assert config.max_new_tokens == 128
    assert config.d_model == 64
    assert config.n_layers == 2
    assert config.n_heads == 2


def test_split_lessons_is_deterministic() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=2, verify=False)

    first_train, first_eval = split_lessons(lessons, seed=7)
    second_train, second_eval = split_lessons(lessons, seed=7)

    assert [lesson.id for lesson in first_train] == [
        lesson.id for lesson in second_train
    ]
    assert [lesson.id for lesson in first_eval] == [
        lesson.id for lesson in second_eval
    ]
    assert first_train
    assert first_eval


def test_tiny_comparison_runs_on_tiny_subset_and_returns_required_keys() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=2, verify=False)[:6]
    config = TinyExperimentConfig(
        batch_size=2,
        train_steps=1,
        router_train_steps=1,
        eval_batches=1,
        max_seq_len=256,
        d_model=32,
        n_layers=1,
        n_heads=2,
        router_hidden_dim=64,
    )

    results = run_tiny_comparison(lessons, config)

    assert [result["routing"] for result in results] == [
        "none",
        "oracle",
        "learned_router",
    ]
    for result in results:
        assert REQUIRED_RESULT_KEYS.issubset(result)
        assert result["finite"] is True
        assert math.isfinite(result["train_loss_last"])
        assert math.isfinite(result["eval_loss_mean"])


def test_tiny_comparison_output_json_and_csv_are_valid(tmp_path) -> None:
    results = [
        {
            "model": "tiny_dense",
            "routing": "none",
            "train_loss_last": 1.0,
            "eval_loss_mean": 1.1,
            "finite": True,
            "train_examples": 4,
            "eval_examples": 2,
        }
    ]

    json_path, csv_path = write_results(results, tmp_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded == results
    assert csv_path is not None
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "model" in csv_text
    assert "tiny_dense" in csv_text


def test_tiny_comparison_can_include_generation_metrics() -> None:
    lessons = generate_coding_bugfix_lessons(count_per_category=2, verify=False)[:6]
    config = TinyExperimentConfig(
        batch_size=2,
        train_steps=1,
        router_train_steps=1,
        eval_batches=1,
        max_seq_len=256,
        d_model=32,
        n_layers=1,
        n_heads=2,
        router_hidden_dim=64,
        run_generation_eval=True,
        generation_eval_examples=1,
        max_new_tokens=2,
    )

    results = run_tiny_comparison(lessons, config)

    for result in results:
        assert "gen_eval_examples" in result
        assert "gen_pass_count" in result
        assert "gen_pass_rate" in result
        assert "gen_failures_by_type" in result


def test_tiny_comparison_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
