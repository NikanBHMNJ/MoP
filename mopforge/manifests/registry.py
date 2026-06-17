"""File-backed registry for planned research run manifests."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.manifests.planner import command_text, dry_run_payload
from mopforge.manifests.run_manifest import ResearchRunManifest


class ManifestRegistry:
    """Local registry rooted at ``manifests/``."""

    def __init__(self, root: str | Path = "manifests") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        if not self.registry_path.exists():
            self._write_registry([])

    def create(self, manifest: ResearchRunManifest) -> ResearchRunManifest:
        directory = self.manifest_dir(manifest.manifest_id)
        directory.mkdir(parents=True, exist_ok=True)
        manifest.save(directory / "manifest.json")
        self.export_command(manifest.manifest_id)
        self.write_dry_run(manifest.manifest_id)
        self._write_registry([item.to_dict() for item in self.list()])
        return manifest

    def load(self, manifest_id: str) -> ResearchRunManifest:
        path = self.manifest_dir(manifest_id) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_id}")
        return ResearchRunManifest.load(path)

    def list(self) -> list[ResearchRunManifest]:
        manifests = []
        for path in self.root.iterdir() if self.root.exists() else []:
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                manifests.append(ResearchRunManifest.load(manifest_path))
        return sorted(manifests, key=lambda item: (item.created_at, item.manifest_id))

    def export_command(self, manifest_id: str) -> Path:
        manifest = self.load(manifest_id)
        path = self.manifest_dir(manifest_id) / "command.sh"
        path.write_text("#!/usr/bin/env sh\n" + command_text(manifest) + "\n", encoding="utf-8")
        return path

    def write_dry_run(self, manifest_id: str) -> Path:
        manifest = self.load(manifest_id)
        path = self.manifest_dir(manifest_id) / "dry_run.json"
        path.write_text(json.dumps(dry_run_payload(manifest), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def manifest_dir(self, manifest_id: str) -> Path:
        return self.root / manifest_id

    def _write_registry(self, records) -> None:
        self.registry_path.write_text(
            json.dumps({"manifests": records}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
