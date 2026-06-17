"""Run a curated CPU-safe subset of MoP-Forge examples."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExampleSpec:
    path: str
    quick: bool = False
    gpu_fallback: bool = False


EXAMPLES = [
    ExampleSpec("examples/runtime_detection.py", quick=True),
    ExampleSpec("examples/gpu_memory_estimate.py", quick=True),
    ExampleSpec("examples/gpu_mop_routing_demo.py", quick=True),
    ExampleSpec("examples/manage_models.py"),
    ExampleSpec("examples/manage_datasets.py"),
    ExampleSpec("examples/analyze_results.py"),
    ExampleSpec("examples/build_paper_report.py"),
    ExampleSpec("examples/gpu_job_profile_validate.py"),
    ExampleSpec("examples/gpu_torchrun_dry_run.py"),
    ExampleSpec("examples/gpu_train_tiny_smoke.py", gpu_fallback=True),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run curated MoP-Forge smoke examples.")
    parser.add_argument("--quick", action="store_true", help="run the tiny quick subset")
    parser.add_argument(
        "--include-gpu-fallback",
        action="store_true",
        help="include the tiny GPUTrainer example, which may run on CPU fallback",
    )
    parser.add_argument("--list", action="store_true", help="list selected examples and exit")
    args = parser.parse_args(argv)

    selected = select_examples(
        quick=bool(args.quick),
        include_gpu_fallback=bool(args.include_gpu_fallback),
    )
    if args.list:
        for spec in selected:
            print(spec.path)
        return 0

    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for spec in selected:
        print(f"RUN {spec.path}")
        completed = subprocess.run(
            [sys.executable, spec.path],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode == 0:
            print(f"PASS {spec.path}")
        else:
            print(f"FAIL {spec.path}")
            print(completed.stdout)
            failures.append(spec.path)
    print(f"summary total={len(selected)} failed={len(failures)}")
    return 1 if failures else 0


def select_examples(*, quick: bool, include_gpu_fallback: bool) -> list[ExampleSpec]:
    selected = [spec for spec in EXAMPLES if (not quick or spec.quick)]
    if include_gpu_fallback:
        for spec in EXAMPLES:
            if spec.gpu_fallback and spec not in selected:
                selected.append(spec)
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
