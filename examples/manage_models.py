"""Register and inspect tiny local model manifests."""

from __future__ import annotations

from mopforge.models import ModelArchitectureConfig, ModelRegistry


def main() -> None:
    print("Local model registry only. It does not train or host models.")
    registry = ModelRegistry("models")
    configs = [
        ModelArchitectureConfig(name="tiny_dense_base", model_type="dense", d_model=32, n_layers=1, n_heads=2, max_seq_len=128),
        ModelArchitectureConfig(name="tiny_mop_oracle", model_type="mop_oracle", d_model=32, n_layers=1, n_heads=2, max_seq_len=128),
        ModelArchitectureConfig(name="tiny_mop_generated", model_type="mop_oracle", d_model=32, n_layers=1, n_heads=2, max_seq_len=128, use_generated_params=True, generated_condition_names=["coding", "debugging", "repair"]),
    ]
    for config in configs:
        manifest = registry.register_model(config)
        print(f"model_ref={manifest.model_id}@{manifest.version_id}")
        print(f"  latest={registry.resolve_model_ref(manifest.model_id).version_id}")
        print(f"  params={manifest.parameter_summary}")


if __name__ == "__main__":
    main()
