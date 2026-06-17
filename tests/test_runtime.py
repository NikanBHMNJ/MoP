"""Tests for Goal 36 runtime/device foundation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopforge.benchmarks import BenchmarkConfig, run_benchmark
from mopforge.cli.main import main as cli_main
from mopforge.configs import (
    MoPForgeConfig,
    dry_run_config,
    get_default_config,
    runtime_config_from_envelope,
    validate_config_envelope,
)
from mopforge.kts import IndexedLessonStore, KnowledgeLesson
from mopforge.pretrain import (
    ContinuedPretrainConfig,
    TextCorpusStore,
    build_demo_code_corpus,
    run_continued_pretraining,
)
from mopforge.runtime import (
    RuntimeConfig,
    autocast_context,
    build_runtime_context,
    detect_devices,
    move_batch_to_device,
    move_model_to_runtime,
    resolve_device,
    resolve_precision,
    runtime_metadata,
)
from mopforge.sft import FinetuneConfig, run_finetune, trainer_config_from_finetune_config
from mopforge.training import TinyTrainer, TrainerConfig


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


def test_runtime_config_validation_roundtrip_and_save_load(tmp_path) -> None:
    config = RuntimeConfig(device="auto", precision="auto", enable_amp=True, require_device_available=False)
    loaded = RuntimeConfig.from_dict(config.to_dict())
    path = config.save(tmp_path / "runtime.json")

    assert loaded == config
    assert RuntimeConfig.load(path) == config
    with pytest.raises(ValueError, match="device"):
        RuntimeConfig(device="")
    with pytest.raises(ValueError, match="precision"):
        RuntimeConfig(precision="float64")


def test_detect_and_resolve_devices_are_cpu_safe() -> None:
    detected = detect_devices()
    json.dumps(detected)
    assert detected["cpu_available"] is True
    assert resolve_device("cpu").selected == "cpu"
    assert resolve_device("auto").selected in {"cpu", "mps", "cuda:0"}
    fallback = resolve_device("cuda", require_available=False)
    assert fallback.selected.startswith(("cpu", "cuda"))
    if not detected["cuda_available"]:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda", require_available=True)


def test_precision_policy_cpu_and_fp8_fallback() -> None:
    cpu = resolve_device("cpu")

    assert resolve_precision("fp32", cpu).selected == "fp32"
    assert resolve_precision("auto", cpu).selected == "fp32"
    fp8 = resolve_precision("fp8", cpu)
    assert fp8.selected == "fp32"
    assert fp8.fp8_requested is True
    assert fp8.warnings


def test_runtime_context_metadata_and_autocast_cpu() -> None:
    runtime = build_runtime_context(RuntimeConfig(device="cpu", precision="fp32"))
    metadata = runtime_metadata(runtime)
    json.dumps(metadata)

    assert metadata["selected_device"] == "cpu"
    assert metadata["selected_precision"] == "fp32"
    with autocast_context(runtime):
        assert True


def test_move_batch_preserves_metadata_and_moves_tensors() -> None:
    torch = pytest.importorskip("torch")
    batch = {
        "input_ids": torch.tensor([1, 2]),
        "nested": [torch.tensor([3]), {"label": "keep"}],
        "target_modules": ["coding", "debugging"],
        "text": "unchanged",
    }

    moved = move_batch_to_device(batch, "cpu")

    assert moved["input_ids"].device.type == "cpu"
    assert moved["nested"][0].device.type == "cpu"
    assert moved["target_modules"] == ["coding", "debugging"]
    assert moved["text"] == "unchanged"


def test_move_model_to_runtime_cpu() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    runtime = build_runtime_context(RuntimeConfig(device="cpu"))
    moved = move_model_to_runtime(model, runtime)

    assert next(moved.parameters()).device.type == "cpu"


def test_tiny_trainer_runtime_path_and_checkpoint_metadata(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = TinyTrainer(
        TrainerConfig(
            run_name="runtime_test",
            lesson_path=str(tmp_path / "lessons.jsonl"),
            index_path=str(tmp_path / "lessons.sqlite"),
            run_registry_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
            device="auto",
            precision="auto",
            enable_amp=True,
            require_device_available=False,
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=64,
            d_model=16,
            n_layers=1,
            n_heads=2,
            save_full_checkpoints=True,
            checkpoint_every_steps=1,
        )
    ).train()

    assert result.metrics["runtime"]["selected_device"] == "cpu"
    assert result.final_state["runtime_metadata"]["selected_precision"] == "fp32"
    checkpoint_path = result.artifacts["full_checkpoint_paths"][-1]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["runtime"]["selected_device"] == "cpu"


def test_sft_runtime_mapping_and_result(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    config = FinetuneConfig(
        lesson_path=str(tmp_path / "lessons.jsonl"),
        index_path=str(tmp_path / "lessons.sqlite"),
        run_registry_root=str(tmp_path / "runs"),
        artifact_root=str(tmp_path / "artifacts"),
        device="auto",
        precision="auto",
        enable_amp=True,
        require_device_available=False,
        max_steps=1,
        eval_batches=1,
        batch_size=1,
        max_seq_len=64,
        save_checkpoints=False,
    )
    trainer_config = trainer_config_from_finetune_config(config)
    result = run_finetune(config)

    assert trainer_config.device == "auto"
    assert trainer_config.precision == "auto"
    assert result.metrics["runtime"]["selected_device"] == "cpu"
    assert Path(result.artifacts["finetune_result_json"]).exists()


def test_cpt_runtime_path(tmp_path) -> None:
    pytest.importorskip("torch")
    corpus_path = tmp_path / "corpus.jsonl"
    TextCorpusStore(corpus_path).add_many(build_demo_code_corpus(count=3))

    result = run_continued_pretraining(
        ContinuedPretrainConfig(
            corpus_path=str(corpus_path),
            lesson_path=None,
            run_registry_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
            device="auto",
            precision="auto",
            enable_amp=True,
            require_device_available=False,
            max_steps=1,
            eval_batches=1,
            batch_size=1,
            max_seq_len=64,
            d_model=16,
            n_layers=1,
            n_heads=2,
        )
    )

    assert result.metrics["runtime"]["selected_device"] == "cpu"
    assert Path(result.artifacts["continued_pretrain_result_json"]).exists()


def test_benchmark_runtime_metadata(tmp_path) -> None:
    pytest.importorskip("torch")
    build_tiny_store(tmp_path)
    result = run_benchmark(
        BenchmarkConfig(
            name="runtime_benchmark",
            benchmark_type="loss",
            lesson_path=str(tmp_path / "lessons.jsonl"),
            index_path=str(tmp_path / "lessons.sqlite"),
            output_root=str(tmp_path / "benchmarks"),
            device="auto",
            precision="auto",
            enable_amp=True,
            require_device_available=False,
            max_examples=2,
            batch_size=1,
            max_seq_len=64,
        )
    )

    assert result.status == "completed"
    assert result.metrics["runtime"]["selected_device"] == "cpu"
    assert result.metrics["runtime.selected_device"] == "cpu"


def test_config_validation_and_dry_run_runtime_fields(tmp_path) -> None:
    trainer = get_default_config("trainer_runtime_auto")
    runtime = get_default_config("runtime_cuda_bf16_plan")
    benchmark = get_default_config("benchmark_runtime_auto")

    assert validate_config_envelope(trainer) == []
    assert validate_config_envelope(runtime) == []
    assert validate_config_envelope(benchmark) == []
    assert dry_run_config(trainer)["runtime"]["selected_device"] == "cpu"
    assert dry_run_config(runtime)["runtime"]["selected_device"] == "cpu"
    assert runtime_config_from_envelope(runtime).require_device_available is False

    path = tmp_path / "trainer_runtime_auto.json"
    trainer.save(path)
    loaded = MoPForgeConfig.load(path)
    assert dry_run_config(loaded)["runtime"]["selected_precision"] == "fp32"


def test_runtime_cli_and_write_defaults(tmp_path, capsys) -> None:
    assert cli_main(["runtime", "detect"]) == 0
    assert "cpu_available=" in capsys.readouterr().out
    assert cli_main(["runtime", "dry-run", "--device", "cpu", "--precision", "fp32"]) == 0
    assert "selected_device=cpu" in capsys.readouterr().out
    assert cli_main([
        "runtime",
        "dry-run",
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--no-require-available",
    ]) == 0
    assert "requested_device=cuda" in capsys.readouterr().out
    assert cli_main(["config", "write-default", "runtime_cpu", str(tmp_path / "runtime_cpu.json")]) == 0
    assert cli_main(["config", "write-default", "trainer_runtime_auto", str(tmp_path / "trainer_runtime_auto.json")]) == 0


def test_optional_cuda_smoke_if_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = resolve_device("cuda")
    policy = resolve_precision("bf16", device, enable_amp=True)

    assert device.selected == "cuda:0"
    assert policy.selected in {"bf16", "fp16"}


def test_runtime_does_not_require_cuda() -> None:
    try:
        import torch
    except Exception:
        return

    assert torch.cuda.is_available() is False
