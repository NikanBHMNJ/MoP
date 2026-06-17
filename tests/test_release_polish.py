from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mopforge
from mopforge.public_api import (
    EXPERIMENTAL_PUBLIC_API,
    PUBLIC_API_POLICY,
    STABLE_PUBLIC_API,
    GPUTrainingConfig,
    KnowledgeLesson,
    RuntimeConfig,
)


def test_package_import_version_and_public_api() -> None:
    assert mopforge.__version__ == "0.45.0"
    assert KnowledgeLesson.__name__ == "KnowledgeLesson"
    assert RuntimeConfig.__name__ == "RuntimeConfig"
    assert GPUTrainingConfig.__name__ == "GPUTrainingConfig"
    assert "KnowledgeLesson" in STABLE_PUBLIC_API
    assert "GPUTrainingConfig" in EXPERIMENTAL_PUBLIC_API
    assert PUBLIC_API_POLICY.to_dict()["stable"]


def test_required_docs_and_release_metadata_exist() -> None:
    required = [
        "README.md",
        "CHANGELOG.md",
        "RELEASE_NOTES_v0.45.0.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/README.md",
        "docs/architecture.md",
        "docs/quickstart.md",
        "docs/api_overview.md",
        "docs/config_templates.md",
        "docs/examples_guide.md",
        "docs/release_checklist.md",
        "docs/known_limitations.md",
        "docs/research_positioning.md",
        "docs/installation.md",
        "docs/command_cookbook.md",
        "docs/serious_jobs_checklist.md",
    ]
    for path in required:
        assert Path(path).exists(), path

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "0.45.0" in readme
    assert "v1.0-beta" in readme
    assert "not a production distributed LLM training framework" in readme


def test_release_check_script_and_json_output(tmp_path) -> None:
    output = tmp_path / "release_check.json"
    completed = subprocess.run(
        [sys.executable, "scripts/release_check.py", "--json", str(output)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True


def test_smoke_example_runner_list_and_quick() -> None:
    listed = subprocess.run(
        [sys.executable, "scripts/run_smoke_examples.py", "--quick", "--list"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert listed.returncode == 0, listed.stdout
    assert "examples/runtime_detection.py" in listed.stdout

    quick = subprocess.run(
        [sys.executable, "scripts/run_smoke_examples.py", "--quick"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert quick.returncode == 0, quick.stdout
    assert "summary total=" in quick.stdout
