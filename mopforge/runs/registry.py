"""File-backed registry for tiny training runs."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.runs.schema import TrainingRunRecord


class RunRegistry:
    """Small file-backed run registry under ``runs/<run_id>/``."""

    def __init__(self, root: str | Path = "runs") -> None:
        """Create a registry rooted at ``root``."""

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        """Return the directory path for a run ID."""

        return self.root / run_id

    def create_run_dir(self, run_id: str) -> Path:
        """Create and return the directory for a run ID."""

        path = self.run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, record: TrainingRunRecord) -> Path:
        """Save ``run.json`` and ``metrics.json`` for a run."""

        run_dir = self.create_run_dir(record.run_id)
        run_path = run_dir / "run.json"
        metrics_path = run_dir / "metrics.json"
        record.artifacts.update(
            {
                "run_json": str(run_path),
                "metrics_json": str(metrics_path),
            }
        )
        metrics_path.write_text(
            json.dumps(record.metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        record.save_json(run_path)
        return run_path

    def list_runs(self) -> list[str]:
        """Return run IDs with a saved ``run.json``."""

        return sorted(
            path.name for path in self.root.iterdir() if (path / "run.json").exists()
        )

    def load(self, run_id: str) -> TrainingRunRecord:
        """Load a saved run by ID."""

        return TrainingRunRecord.load_json(self.run_dir(run_id) / "run.json")
