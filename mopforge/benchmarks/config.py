"""Benchmark config schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mopforge.configs.io import MoPForgeConfig
from mopforge.runtime import RuntimeConfig


BENCHMARK_TYPES = {
    "loss",
    "code_correctness",
    "router",
    "parameter_efficiency",
    "composite",
}


@dataclass(slots=True)
class BenchmarkConfig:
    """CPU-safe local benchmark configuration."""

    name: str
    benchmark_type: str = "loss"
    description: str = ""
    model_type: str = "dense"
    model_ref: str | None = None
    checkpoint_path: str | None = None
    run_id: str | None = None

    lesson_path: str = "data/indexed_lessons.jsonl"
    index_path: str = "data/kts_index.sqlite"
    corpus_path: str | None = None
    dataset_ref: str | None = None
    dataset_split: str | None = None

    max_examples: int = 8
    batch_size: int = 2
    max_seq_len: int = 512
    seed: int = 123
    device: str = "cpu"
    precision: str = "fp32"
    enable_amp: bool = False
    allow_tf32: bool = False
    deterministic: bool = False
    compile_model: bool = False
    require_device_available: bool = True

    tokenizer_type: str = "byte"
    tokenizer_name_or_path: str | None = None
    tokenizer_spec_path: str | None = None

    generation_max_new_tokens: int = 64
    generation_examples: int = 4

    target_modules: list[str] | None = None
    use_fast_adapters: bool = False
    use_generated_params: bool = False
    generated_condition_names: list[str] | None = None

    output_root: str = "benchmarks"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _require_non_empty(self.name, "name")
        if self.benchmark_type not in BENCHMARK_TYPES:
            valid = ", ".join(sorted(BENCHMARK_TYPES))
            raise ValueError(f"benchmark_type must be one of: {valid}.")
        if self.model_type not in {"dense", "mop_oracle", "mop_learned_router"}:
            raise ValueError("model_type must be dense, mop_oracle, or mop_learned_router.")
        for field_name in (
            "max_examples",
            "batch_size",
            "max_seq_len",
            "generation_max_new_tokens",
            "generation_examples",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer.")
        RuntimeConfig(
            device=self.device,
            precision=self.precision,
            enable_amp=self.enable_amp,
            allow_tf32=self.allow_tf32,
            deterministic=self.deterministic,
            compile_model=self.compile_model,
            require_device_available=self.require_device_available,
        )
        for field_name in (
            "checkpoint_path",
            "run_id",
            "model_ref",
            "lesson_path",
            "index_path",
            "corpus_path",
            "dataset_ref",
            "dataset_split",
            "tokenizer_name_or_path",
            "tokenizer_spec_path",
            "output_root",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string or None.")
            setattr(self, field_name, value.strip())
        if self.checkpoint_path is not None and not Path(self.checkpoint_path).exists():
            raise ValueError(f"checkpoint_path does not exist: {self.checkpoint_path}")
        self.target_modules = _normalize_strings(self.target_modules, "target_modules")
        self.generated_condition_names = _normalize_strings(
            self.generated_condition_names,
            "generated_condition_names",
        )
        if not isinstance(self.tokenizer_type, str) or not self.tokenizer_type.strip():
            raise ValueError("tokenizer_type must be a non-empty string.")
        self.tokenizer_type = self.tokenizer_type.strip().lower()
        if self.tokenizer_type == "hf" and self.tokenizer_name_or_path is None and self.tokenizer_spec_path is None:
            raise ValueError("HF tokenizer configs require tokenizer_name_or_path or tokenizer_spec_path.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Create a benchmark config from a dictionary."""

        if not isinstance(data, dict):
            raise TypeError("BenchmarkConfig.from_dict expects a dictionary.")
        return cls(**dict(data))

    def save(self, path: str | Path) -> Path:
        """Save this config as a benchmark envelope."""

        return MoPForgeConfig(kind="benchmark", payload=self.to_dict()).save(path)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        """Load a benchmark config from a benchmark envelope file."""

        envelope = MoPForgeConfig.load(path)
        if envelope.kind != "benchmark":
            raise ValueError(f"Expected kind='benchmark', got {envelope.kind!r}.")
        return cls.from_dict(envelope.payload)


def _normalize_strings(values: list[str] | None, field_name: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be a list of strings.")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    seen = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
