"""Tests for local result importer."""

from __future__ import annotations

import json
from pathlib import Path

from mopforge.cli.main import main as cli_main
from mopforge.importers import ResultImportConfig, ResultImportRegistry, detect_artifacts, import_results


def write_run(path: Path) -> Path:
    path.mkdir(parents=True)
    result = {"run_id": "run-a", "model_type": "dense", "metrics": {"eval_loss_mean": 0.1, "finite": True}}
    output = path / "trainer_result.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    return output


def test_import_config_detection_and_run_result(tmp_path) -> None:
    write_run(tmp_path / "remote")
    assert detect_artifacts(tmp_path / "remote")
    record = import_results(ResultImportConfig(name="demo", source_path=str(tmp_path / "remote"), output_root=str(tmp_path / "imports")))
    assert record.status == "completed"
    rows = json.loads(Path(record.normalized_results_path).read_text(encoding="utf-8"))
    assert rows[0]["run_id"] == "run-a"
    assert ResultImportRegistry(tmp_path / "imports").load_record(record.import_id).import_id == record.import_id


def test_import_cli(tmp_path, capsys) -> None:
    write_run(tmp_path / "remote")
    root = tmp_path / "imports"
    assert cli_main(["import", "results", str(tmp_path / "remote"), "--name", "demo", "--root", str(root)]) == 0
    output = capsys.readouterr().out
    import_id = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("import_id=")][0]
    assert cli_main(["import", "list", "--root", str(root)]) == 0
    assert import_id in capsys.readouterr().out
    assert cli_main(["import", "show", import_id, "--root", str(root)]) == 0
