"""Build the committed Goal 50 Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Goal 50: 100M Code Learning Gate on Colab L4

This notebook diagnoses the Goal 49 zero-verifier result before any 1B run. It trains one 100M Dense model on a balanced 50-lesson verified repair dataset, counts **optimizer updates** separately from microsteps, evaluates full train and held-out generation from the best checkpoint, and creates a lightweight pass/fail report.

No quantization is used. A failed gate means the protocol, optimization budget, or generation path still needs work; it is not evidence that 100M capacity is the blocker.
"""
    ),
    markdown(
        """## 0. Settings

Select **Runtime > Change runtime type > GPU > L4**. The acceptance thresholds are protocol gates from `AGENTS.md`; do not lower them to manufacture a pass.
"""
    ),
    code(
        """REPO_URL = "https://github.com/NikanBHMNJ/MoP.git"
REPO_DIR = "/content/MoP"
REPORT_ID = "goal50_100m_learning_gate"
REPORT_DIR = f"reports/{REPORT_ID}"

COUNT_PER_CATEGORY = 10
SPLIT_SEED = 42
TRAIN_SHUFFLE_SEED = 42
OPTIMIZER_UPDATES = 1000
GRADIENT_ACCUMULATION_STEPS = 8
MAX_STEPS = OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION_STEPS
EVAL_EVERY_OPTIMIZER_UPDATES = 100
EVAL_EVERY_STEPS = EVAL_EVERY_OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION_STEPS
GENERATION_EVAL_EXAMPLES = 50
GENERATION_MAX_NEW_TOKENS = 256
QUALITY_FORMAT = "fixed_code_xml"
REQUIRE_CUDA = True

GATES = {
    "ground_truth_controls": 1.0,
    "train_fixed_code_complete_rate": 0.95,
    "train_syntax_pass_rate": 0.95,
    "train_verifier_pass_rate": 0.95,
    "train_exact_match_rate": 0.90,
}
"""
    ),
    code(
        """from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def sh(command: str, *, check: bool = True) -> str:
    print(f"\\n$ {command}")
    result = subprocess.run(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(result.stdout)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")
    return result.stdout


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output


def parse_key(output: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", output, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"missing {key}=... in command output")
    return match.group(1).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
"""
    ),
    markdown("## 1. Clone, Install, And Verify The L4\n"),
    code(
        """repo_dir = Path(REPO_DIR)
if not (repo_dir / ".git").exists():
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    sh(f"git clone --depth 1 {REPO_URL} {repo_dir}")
else:
    os.chdir(repo_dir)
    sh("git pull --ff-only", check=False)
os.chdir(repo_dir)

sh("python -m pip install -q -e .[dev,gpu]")
sh("mopforge version")
sh("mopforge runtime detect")
sh("nvidia-smi", check=False)

import torch

if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this learning-gate notebook.")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
"""
    ),
    markdown("## 2. Build A Balanced Fixed Verified Split\n"),
    code(
        """dataset_output = sh(
    " ".join(
        [
            "mopforge gpu prepare-efficiency-data",
            f"--dataset-id {REPORT_ID}",
            f"--count-per-category {COUNT_PER_CATEGORY}",
            f"--split-seed {SPLIT_SEED}",
            "--stratify-by bug_type",
            f"--quality-format {QUALITY_FORMAT}",
            "--verify",
            "--overwrite",
        ]
    )
)
DATASET_REF = parse_key(dataset_output, "dataset_ref")
DATASET_SPLIT_ID = parse_key(dataset_output, "split_id")
print("DATASET_REF=", DATASET_REF)
print("DATASET_SPLIT_ID=", DATASET_SPLIT_ID)
"""
    ),
    markdown(
        """## 3. Train The 100M Memorization Diagnostic

`MAX_STEPS` is the microstep budget. The report must show exactly `OPTIMIZER_UPDATES` optimizer updates; this is the distinction Goal 49 obscured.
"""
    ),
    code(
        """template = read_json("configs/jobs/100m_dense_extended_efficiency.json")
payload = dict(template["payload"])
payload.update(
    {
        "name": REPORT_ID,
        "device": "cuda",
        "precision": "bf16",
        "require_device_available": True,
        "dataset_ref": DATASET_REF,
        "dataset_split_id": DATASET_SPLIT_ID,
        "max_steps": MAX_STEPS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "eval_every_steps": EVAL_EVERY_STEPS,
        "eval_full_dataset": True,
        "save_every_steps": MAX_STEPS,
        "save_best_eval_checkpoint": True,
        "shuffle_train": True,
        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "run_generation_eval": True,
        "generation_eval_examples": GENERATION_EVAL_EXAMPLES,
        "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "generation_eval_use_best_checkpoint": True,
        "generation_eval_include_train": True,
        "generation_eval_stratify_by": "bug_type",
        "output_root": "gpu_runs",
        "artifact_root": "artifacts",
    }
)
metadata = dict(payload.get("metadata") or {})
metadata.update(
    {
        "report_id": REPORT_ID,
        "phase": "100m_memorization_diagnostic",
        "optimizer_updates_requested": OPTIMIZER_UPDATES,
        "microsteps_requested": MAX_STEPS,
        "split_seed": SPLIT_SEED,
        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "quality_format": QUALITY_FORMAT,
        "hardware_target": "google_colab_l4",
        "no_quantization": True,
    }
)
payload["metadata"] = metadata
CONFIG_PATH = write_json(
    "configs/jobs/colab_l4_goal50_100m_learning_gate.json",
    {"kind": "gpu_train", "version": template.get("version", "1"), "payload": payload},
)

sh(f"mopforge gpu validate {CONFIG_PATH}")
train_output = sh(f"mopforge gpu train {CONFIG_PATH}")
RUN_ID = parse_key(train_output, "run_id")
RUN_DIR = Path("gpu_runs") / RUN_ID
print("RUN_ID=", RUN_ID)
"""
    ),
    markdown("## 4. Evaluate The Learning Gate\n"),
    code(
        """metrics = read_json(RUN_DIR / "metrics.json")
generation = read_json(RUN_DIR / "generation_eval.json")
controls = read_json(RUN_DIR / "ground_truth_controls.json")
train_summary = generation["splits"]["train"]["summary"]
eval_summary = generation["splits"]["eval"]["summary"]
length_stats = metrics["data"]["sequence_length_statistics"]
train_categories = set(train_summary.get("per_category", {}))
eval_categories = set(eval_summary.get("per_category", {}))

checks = {
    "optimizer_update_budget": metrics["optimizer_steps"] == OPTIMIZER_UPDATES,
    "best_checkpoint_generation": generation["checkpoint"]["source"] == "best_eval",
    "all_five_train_categories": len(train_categories) == 5,
    "all_five_eval_categories": len(eval_categories) == 5,
    "full_eval_loss": metrics["eval_full_dataset"] is True and metrics["latest_eval_examples"] == metrics["data"]["eval_examples"],
    "no_train_or_eval_truncation": length_stats["train"]["truncated_examples"] == 0 and length_stats["eval"]["truncated_examples"] == 0,
    "generation_budget_covers_targets": max(length_stats["train"]["max_target_tokens"], length_stats["eval"]["max_target_tokens"]) <= GENERATION_MAX_NEW_TOKENS,
    "ground_truth_controls": controls["passed"] is True,
    "train_fixed_code_complete_rate": train_summary["gen_fixed_code_complete_rate"] >= GATES["train_fixed_code_complete_rate"],
    "train_syntax_pass_rate": train_summary["gen_syntax_pass_rate"] >= GATES["train_syntax_pass_rate"],
    "train_verifier_pass_rate": train_summary["gen_verifier_pass_rate"] >= GATES["train_verifier_pass_rate"],
    "train_exact_match_rate": train_summary["gen_exact_match_rate"] >= GATES["train_exact_match_rate"],
}
gate_passed = all(checks.values())
gate = {
    "passed": gate_passed,
    "checks": checks,
    "thresholds": GATES,
    "microsteps": metrics["global_steps"],
    "optimizer_updates": metrics["optimizer_steps"],
    "train_epoch": metrics["train_epoch"],
    "checkpoint": generation["checkpoint"],
    "sequence_length_statistics": length_stats,
    "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    "train_summary": train_summary,
    "eval_summary": eval_summary,
}
print(json.dumps(gate, indent=2, sort_keys=True))
print("\\nGOAL 50 MEMORIZATION GATE:", "PASS" if gate_passed else "FAIL")
if not gate_passed:
    print("Do not start a 1B run. Inspect per-category samples and failure types first.")
"""
    ),
    markdown("## 5. Build And Download The Lightweight Report\n"),
    code(
        """REPORT_PATH = Path(REPORT_DIR)
if REPORT_PATH.exists():
    shutil.rmtree(REPORT_PATH)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

for filename in [
    "config.json",
    "gpu_training_result.json",
    "metrics.json",
    "runtime.json",
    "state.json",
    "memory_estimate.json",
    "generation_eval.json",
    "ground_truth_controls.json",
]:
    source = RUN_DIR / filename
    if source.exists():
        shutil.copy2(source, REPORT_PATH / filename)

write_json(REPORT_PATH / "learning_gate.json", gate)
write_json(
    REPORT_PATH / "experiment_settings.json",
    {
        "report_id": REPORT_ID,
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_ref": DATASET_REF,
        "dataset_split_id": DATASET_SPLIT_ID,
        "split_seed": SPLIT_SEED,
        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "count_per_category": COUNT_PER_CATEGORY,
        "microsteps": MAX_STEPS,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "full_eval": True,
        "generation_eval_stratify_by": "bug_type",
        "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "quality_format": QUALITY_FORMAT,
        "hardware_target": "google_colab_l4",
        "quantization": None,
    },
)

readme = f'''# Goal 50 100M Learning Gate

Generated by `notebooks/colab_l4_goal50_100m_learning_gate.ipynb`.

## Result

**Memorization gate: {'PASS' if gate_passed else 'FAIL'}**

- Run: `{RUN_ID}`
- Dataset: `{DATASET_REF}`
- Split: `{DATASET_SPLIT_ID}`
- Microsteps: `{metrics['global_steps']}`
- Optimizer updates: `{metrics['optimizer_steps']}`
- Epochs entered: `{metrics['train_epoch']}`
- Generation checkpoint: `{generation['checkpoint']['path']}` at step `{generation['checkpoint']['global_step']}`
- Train XML completion: `{train_summary['gen_fixed_code_complete_rate']:.2%}`
- Train syntax pass: `{train_summary['gen_syntax_pass_rate']:.2%}`
- Train verifier pass: `{train_summary['gen_verifier_pass_rate']:.2%}`
- Train exact match: `{train_summary['gen_exact_match_rate']:.2%}`
- Held-out verifier pass: `{eval_summary['gen_verifier_pass_rate']:.2%}`
- Ground-truth controls: `{'PASS' if controls['passed'] else 'FAIL'}`

The report contains generated samples and per-category failure evidence. It excludes checkpoints, caches, optimizer state, and model weights. A failed gate blocks the 1B run; a pass permits the larger 100M comparison, but does not itself prove generalization.
'''
(REPORT_PATH / "README.md").write_text(readme, encoding="utf-8")

forbidden_suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
forbidden = [
    str(path.relative_to(REPORT_PATH))
    for path in REPORT_PATH.rglob("*")
    if path.is_file() and path.suffix.lower() in forbidden_suffixes
]
if forbidden:
    raise RuntimeError(f"model/checkpoint files entered report: {forbidden}")

write_json(
    REPORT_PATH / "report_manifest.json",
    {
        "report_id": REPORT_ID,
        "forbidden_model_files_detected": False,
        "files": [
            {
                "path": str(path.relative_to(REPORT_PATH)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(REPORT_PATH.rglob("*"))
            if path.is_file()
        ],
    },
)

zip_path = Path(
    shutil.make_archive(
        str(Path("reports") / REPORT_ID),
        "zip",
        root_dir=REPORT_PATH.parent,
        base_dir=REPORT_PATH.name,
    )
)
print((REPORT_PATH / "README.md").read_text(encoding="utf-8"))
print("Report ZIP:", zip_path)

try:
    from google.colab import files

    files.download(str(zip_path))
except ImportError:
    print("Not running in Colab; the ZIP remains at", zip_path)
"""
    ),
    markdown(
        """## 6. Scaling Decision

- **FAIL:** inspect `generation_eval.json` by category and failure type; do not run 1B.
- **PASS with weak held-out quality:** expand to at least 10,000 balanced verified lessons and run the full Goal 50 100M comparison.
- **PASS with improving held-out quality:** the pipeline has demonstrated learning; only then run a short 1B L4 memory/throughput probe.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "colab_l4_goal50_100m_learning_gate.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path("notebooks/colab_l4_goal50_100m_learning_gate.ipynb")
output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(output)
