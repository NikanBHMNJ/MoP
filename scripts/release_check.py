"""Lightweight release readiness checks for MoP-Forge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass, asdict
from pathlib import Path


REQUIRED_VERSION = "0.46.0"
REQUIRED_DOCS = [
    "docs/README.md",
    "docs/architecture.md",
    "docs/quickstart.md",
    "docs/gpu_quickstart.md",
    "docs/colab_training.md",
    "docs/gpu_job_profiles.md",
    "docs/gpu_efficiency_benchmarking.md",
    "docs/gpu_runtime_limitations.md",
    "docs/serious_jobs_checklist.md",
    "docs/config_templates.md",
    "docs/examples_guide.md",
    "docs/api_overview.md",
    "docs/release_checklist.md",
    "docs/known_limitations.md",
    "docs/research_positioning.md",
    "docs/installation.md",
    "docs/command_cookbook.md",
]


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight MoP-Forge release checks.")
    parser.add_argument("--quick-examples", action="store_true", help="run quick smoke examples")
    parser.add_argument("--json", dest="json_path", help="write check results as JSON")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    checks = run_checks(root, quick_examples=bool(args.quick_examples))
    payload = {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}")
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote_json={output}")
    return 0 if payload["ok"] else 1


def run_checks(root: Path, *, quick_examples: bool = False) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.append(_check_import_version())
    checks.append(_check_pyproject_version(root))
    checks.append(_check_cli(root, ["version"], REQUIRED_VERSION))
    checks.append(_check_cli(root, ["runtime", "detect"], "torch_available="))
    checks.append(_check_cli(root, ["gpu", "validate", "configs/jobs/tiny_gpu_smoke.json"], "validation=valid"))
    checks.append(_check_cli(root, ["gpu", "estimate", "configs/jobs/100m_mop_a100_smoke.json"], "total_memory_gb_estimate"))
    checks.append(_check_docs(root))
    if quick_examples:
        checks.append(
            _check_subprocess(
                root,
                [sys.executable, "scripts/run_smoke_examples.py", "--quick"],
                "quick_examples",
                "summary",
            )
        )
    return checks


def _check_import_version() -> CheckResult:
    try:
        import mopforge
        from mopforge.public_api import PUBLIC_API_POLICY

        ok = mopforge.__version__ == REQUIRED_VERSION and bool(PUBLIC_API_POLICY.stable)
        return CheckResult("import_version", ok, mopforge.__version__)
    except Exception as exc:
        return CheckResult("import_version", False, str(exc))


def _check_pyproject_version(root: Path) -> CheckResult:
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        version = data["project"]["version"]
        return CheckResult("pyproject_version", version == REQUIRED_VERSION, version)
    except Exception as exc:
        return CheckResult("pyproject_version", False, str(exc))


def _check_docs(root: Path) -> CheckResult:
    missing = [path for path in REQUIRED_DOCS if not (root / path).exists()]
    if missing:
        return CheckResult("required_docs", False, "missing=" + ",".join(missing))
    return CheckResult("required_docs", True, f"count={len(REQUIRED_DOCS)}")


def _check_cli(root: Path, args: list[str], expected: str) -> CheckResult:
    return _check_subprocess(root, [sys.executable, "-m", "mopforge.cli.main", *args], "cli_" + "_".join(args[:2]), expected)


def _check_subprocess(root: Path, command: list[str], name: str, expected: str) -> CheckResult:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout
    ok = completed.returncode == 0 and expected in output
    detail = f"returncode={completed.returncode}"
    if not ok:
        detail += " output=" + output[-500:]
    return CheckResult(name, ok, detail)


if __name__ == "__main__":
    raise SystemExit(main())
