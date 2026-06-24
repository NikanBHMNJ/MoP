from pathlib import Path
import json
import os
import socket
import subprocess
import sys

import pytest

from mopforge.gpu import (
    DistributedConfig,
    DistributedRuntime,
    build_torchrun_command,
    consolidate_sharded_gpu_checkpoint,
    launch_torchrun_dry_run,
    load_sharded_training_checkpoint,
    save_sharded_training_checkpoint,
)
from mopforge.models import ModelArchitectureConfig, build_tiny_model_from_architecture
from mopforge.tokenization import ByteTokenizer


def test_distributed_config_and_torchrun_dry_run_command():
    config = DistributedConfig(strategy="torchrun", nproc_per_node=2, dry_run=True)
    command = build_torchrun_command("config.json", config)
    assert command[0] == "torchrun"
    assert "--nproc_per_node" in command
    payload = launch_torchrun_dry_run("config.json", config)
    assert payload["executes"] is False
    assert payload["command"][0] == "torchrun"


def test_sharded_checkpoint_roundtrip_without_process_group(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 4)
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()
    expected = model(inputs).detach().clone()
    runtime = DistributedRuntime()

    path = save_sharded_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        trainer_state={"global_step": 1},
        config={"name": "test"},
        runtime=runtime,
    )
    with torch.no_grad():
        model.weight.zero_()
    payload = load_sharded_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        runtime=runtime,
    )

    assert torch.allclose(model(inputs), expected)
    assert payload["trainer_state"]["global_step"] == 1
    assert Path(path, "manifest.json").exists()


def test_sharded_checkpoint_consolidates_to_standard_gpu_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    tokenizer = ByteTokenizer()
    architecture = ModelArchitectureConfig(
        name="consolidate-smoke",
        architecture_family="production_decoder_v2",
        model_type="dense",
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_key_value_heads=1,
        intermediate_size=32,
        max_seq_len=16,
    )
    model = build_tiny_model_from_architecture(architecture, tokenizer=tokenizer)
    from mopforge.gpu import GPUTrainingConfig

    config = GPUTrainingConfig(
        name="consolidate-smoke",
        architecture_family="production_decoder_v2",
        tokenizer_type="byte",
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_key_value_heads=1,
        intermediate_size=32,
        max_seq_len=16,
        device="cpu",
        precision="fp32",
        enable_amp=False,
        distributed_strategy="fsdp",
        distributed_checkpoint_mode="sharded",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    expected = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    source = save_sharded_training_checkpoint(
        tmp_path / "sharded",
        model=model,
        optimizer=optimizer,
        trainer_state={"global_step": 7, "optimizer_step": 2},
        config=config,
        runtime=DistributedRuntime(),
        metadata={"model": {"architecture": architecture.to_dict()}},
    )

    result = consolidate_sharded_gpu_checkpoint(source, tmp_path / "full.pt")

    payload = torch.load(result["output"], map_location="cpu", weights_only=False)
    assert payload["trainer_state"]["global_step"] == 7
    assert payload["optimizer_state"] is None
    assert payload["model_metadata"]["architecture"]["d_model"] == 16
    assert all(
        torch.equal(payload["model_state"][name], value)
        for name, value in expected.items()
    )


def test_two_process_ddp_training_writes_one_result_and_sharded_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    lessons = tmp_path / "lessons.jsonl"
    records = []
    for index in range(8):
        records.append(
            {
                "id": f"ddp-{index}",
                "domain": "coding",
                "skill": "completion",
                "subskill": "return",
                "difficulty": 1,
                "target_modules": ["coding"],
                "input": f"def value_{index}():\n    pass",
                "expected_output": f"def value_{index}():\n    return {index}",
                "verification": {"type": "tests", "status": "verified"},
                "metadata": {},
            }
        )
    lessons.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_root = tmp_path / "gpu_runs"
    artifact_root = tmp_path / "artifacts"
    config_path = tmp_path / "ddp.json"
    config_path.write_text(
        json.dumps(
            {
                "kind": "gpu_train",
                "version": "1",
                "payload": {
                    "name": "ddp-integration",
                    "architecture_family": "production_decoder_v2",
                    "model_type": "dense",
                    "lesson_path": str(lessons),
                    "output_root": str(output_root),
                    "artifact_root": str(artifact_root),
                    "tokenizer_type": "byte",
                    "max_steps": 1,
                    "max_optimizer_steps": 1,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "eval_every_optimizer_steps": 1,
                    "eval_batches": 1,
                    "save_every_optimizer_steps": 1,
                    "log_every_optimizer_steps": 1,
                    "d_model": 16,
                    "n_layers": 1,
                    "n_heads": 2,
                    "n_key_value_heads": 1,
                    "intermediate_size": 32,
                    "max_seq_len": 64,
                    "device": "cpu",
                    "precision": "fp32",
                    "enable_amp": False,
                    "require_device_available": True,
                    "distributed_strategy": "ddp",
                    "distributed_backend": "gloo",
                    "distributed_checkpoint_mode": "sharded",
                    "save_full_checkpoints": True,
                    "save_optimizer_state": True,
                    "run_generation_eval": False
                }
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "mopforge.cli.main",
        "gpu",
        "train",
        str(config_path),
    ]
    master_port = _free_port()
    workers = []
    for rank in range(2):
        env = {
            **os.environ,
            "USE_LIBUV": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
        }
        workers.append(
            subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    outputs = [worker.communicate(timeout=120) for worker in workers]
    failures = [
        f"rank={rank}\n{stdout}\n{stderr}"
        for rank, (worker, (stdout, stderr)) in enumerate(zip(workers, outputs))
        if worker.returncode != 0
    ]
    assert not failures, "\n".join(failures)
    registry = json.loads((output_root / "registry.json").read_text(encoding="utf-8"))
    assert len(registry["runs"]) == 1
    run_dir = Path(registry["runs"][0]["output_dir"])
    result = json.loads((run_dir / "gpu_training_result.json").read_text(encoding="utf-8"))
    assert result["state"]["optimizer_step"] == 1
    checkpoints = list((run_dir / "checkpoints").glob("checkpoint-step-*"))
    assert checkpoints
    assert (checkpoints[0] / "manifest.json").is_file()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]
