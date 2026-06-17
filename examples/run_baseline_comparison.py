"""Build a tiny baseline comparison experiment config."""

from __future__ import annotations

from mopforge.baselines import build_baseline_experiment_config, list_baselines


def main() -> None:
    print("Baseline comparison config only. No GPU execution.")
    names = ["dense_full", "adapter_only", "generated_params_only", "mop_module_only", "moe_tiny"]
    experiment = build_baseline_experiment_config(names)
    print(f"baseline_count={len(names)}")
    print(f"available={','.join(spec.name for spec in list_baselines())}")
    print(f"experiment_name={experiment.name}")
    print(f"run_count={len(experiment.runs)}")
    print("moe_tiny_implementation=moe_tiny_shim")


if __name__ == "__main__":
    main()
