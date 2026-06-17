"""Import local result artifacts into normalized rows."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.importers import ResultImportConfig, import_results


def main() -> None:
    print("Local result importer only. It does not fetch remote files.")
    if Path("runs").exists():
        source = "runs"
    elif Path("outputs").exists():
        source = "outputs"
    else:
        demo = Path("outputs/import_demo_source")
        demo.mkdir(parents=True, exist_ok=True)
        (demo / "trainer_result.json").write_text(
            json.dumps({"run_id": "import-demo", "model_type": "dense", "metrics": {"eval_loss_mean": 1.0, "finite": True}}, indent=2),
            encoding="utf-8",
        )
        source = str(demo)
    record = import_results(ResultImportConfig(name="local_results_demo", source_path=source, output_root="imports", copy_files=False))
    print(f"import_id={record.import_id}")
    print(f"artifact_count={record.metadata.get('artifact_count')}")
    print(f"row_count={record.metadata.get('row_count')}")
    print(f"normalized_results_path={record.normalized_results_path}")


if __name__ == "__main__":
    main()
