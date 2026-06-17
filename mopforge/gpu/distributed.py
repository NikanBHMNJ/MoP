"""Torchrun/distributed launch planning for GPU jobs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DistributedConfig:
    strategy: str = "none"
    num_nodes: int = 1
    nproc_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    node_rank: int = 0
    backend: str = "nccl"
    dry_run: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "torchrun", "ddp_plan"}:
            raise ValueError("strategy must be none, torchrun, or ddp_plan.")
        if type(self.num_nodes) is not int or self.num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer.")
        if type(self.nproc_per_node) is not int or self.nproc_per_node <= 0:
            raise ValueError("nproc_per_node must be a positive integer.")
        if type(self.master_port) is not int or not 0 < self.master_port < 65536:
            raise ValueError("master_port must be a valid TCP port.")
        if type(self.node_rank) is not int or self.node_rank < 0:
            raise ValueError("node_rank must be a non-negative integer.")
        if self.backend not in {"nccl", "gloo"}:
            raise ValueError("backend must be nccl or gloo.")
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be a boolean.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistributedConfig":
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "DistributedConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_torchrun_command(config_path: str, distributed: DistributedConfig | dict[str, Any] | None = None) -> list[str]:
    cfg = distributed if isinstance(distributed, DistributedConfig) else DistributedConfig.from_dict(distributed or {"strategy": "torchrun"})
    if cfg.strategy == "none":
        return ["mopforge", "gpu", "train", config_path]
    return [
        "torchrun",
        "--nnodes",
        str(cfg.num_nodes),
        "--nproc_per_node",
        str(cfg.nproc_per_node),
        "--master_addr",
        cfg.master_addr,
        "--master_port",
        str(cfg.master_port),
        "--node_rank",
        str(cfg.node_rank),
        "-m",
        "mopforge.cli.main",
        "gpu",
        "train",
        config_path,
    ]


def validate_distributed_plan(distributed: DistributedConfig, execute: bool = False) -> list[str]:
    messages: list[str] = []
    if execute and distributed.backend == "nccl":
        try:
            import torch

            if not torch.cuda.is_available():
                messages.append("ERROR: nccl execution requires CUDA.")
        except Exception:
            messages.append("ERROR: nccl execution requires PyTorch CUDA.")
    if distributed.strategy in {"torchrun", "ddp_plan"} and distributed.dry_run:
        messages.append("WARNING: torchrun launcher is in dry-run planning mode.")
    return messages
