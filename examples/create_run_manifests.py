"""Create local run manifests for future GPU jobs without executing them."""

from __future__ import annotations

from mopforge.configs import default_sft_config
from mopforge.manifests import ManifestRegistry, ResourceSpec, command_text, plan_run_manifest


def main() -> None:
    print("Run manifests are plans only. No GPU job is launched.")
    config = default_sft_config("sft_full")
    manifest = plan_run_manifest(config, ResourceSpec(accelerator="cpu"), name="example_cpu_sft")
    registry = ManifestRegistry("manifests")
    registry.create(manifest)
    print(f"manifest_id={manifest.manifest_id}")
    print(f"command={command_text(manifest)}")
    print(f"dry_run_path={registry.manifest_dir(manifest.manifest_id) / 'dry_run.json'}")


if __name__ == "__main__":
    main()
