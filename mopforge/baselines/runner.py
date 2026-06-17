"""Build baseline experiment configs."""

from __future__ import annotations

from mopforge.baselines.catalog import get_baseline
from mopforge.configs import default_sft_config
from mopforge.experiments.matrix import ExperimentConfig


def build_baseline_experiment_config(
    baseline_names: list[str],
    *,
    name: str = "baseline_comparison_cpu",
) -> ExperimentConfig:
    """Create a tiny list experiment comparing catalog baselines."""

    runs = []
    for baseline_name in baseline_names:
        spec = get_baseline(baseline_name)
        if spec.name == "dense_full":
            envelope = default_sft_config("sft_full")
        elif spec.use_fast_adapters:
            envelope = default_sft_config("sft_adapter")
        elif spec.use_generated_params:
            envelope = default_sft_config("sft_generated")
        else:
            envelope = default_sft_config("sft_module" if spec.model_type != "dense" else "sft_full")
        payload = dict(envelope.payload)
        payload["model_type"] = spec.model_type
        payload["use_fast_adapters"] = spec.use_fast_adapters
        payload["use_generated_params"] = spec.use_generated_params
        payload["max_steps"] = 1
        payload["eval_batches"] = 1
        payload["batch_size"] = 1
        payload["max_seq_len"] = min(int(payload.get("max_seq_len", 128)), 128)
        envelope.payload = payload
        envelope.metadata.update({"baseline": spec.to_dict(), "baseline_name": baseline_name})
        runs.append(envelope.to_dict())
    return ExperimentConfig(
        name=name,
        kind="list",
        description="Tiny CPU baseline comparison.",
        runs=runs,
        max_runs=len(runs),
        tags=["baseline", "cpu", "smoke"],
    )
