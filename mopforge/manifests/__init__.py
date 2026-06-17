"""Research run manifests for future GPU/cloud jobs."""

from mopforge.manifests.planner import (
    command_text,
    config_from_path_or_payload,
    dry_run_payload,
    plan_run_manifest,
)
from mopforge.manifests.resources import ACCELERATORS, PRECISIONS, ResourceSpec
from mopforge.manifests.registry import ManifestRegistry
from mopforge.manifests.run_manifest import (
    ManifestConfig,
    ResearchRunManifest,
)

__all__ = [
    "ACCELERATORS",
    "PRECISIONS",
    "ManifestConfig",
    "ResearchRunManifest",
    "ResourceSpec",
    "ManifestRegistry",
    "command_text",
    "config_from_path_or_payload",
    "dry_run_payload",
    "plan_run_manifest",
]
