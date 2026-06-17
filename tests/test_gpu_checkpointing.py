from pathlib import Path

import pytest

from tests.test_gpu_trainer import _config
from mopforge.gpu import GPUTrainer, GPURunRegistry


pytest.importorskip("torch")


def test_gpu_run_registry_latest_checkpoint_and_show_record(tmp_path):
    result = GPUTrainer(_config(tmp_path)).train()
    registry = GPURunRegistry(tmp_path / "gpu_runs")
    record = registry.load_record(result.run_id)
    assert record.latest_checkpoint_path == result.artifacts["latest_checkpoint_path"]
    assert Path(registry.latest_checkpoint(result.run_id)).exists()
