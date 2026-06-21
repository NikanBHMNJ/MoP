"""Build the gated Goal 50 full 100M comparison notebook."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("notebooks/colab_l4_goal49_verified_code_quality_report.ipynb")
OUTPUT = Path("notebooks/colab_l4_goal50_100m_quality_comparison.ipynb")


def lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0]["source"] = lines(
    """# Goal 50 Full 100M Verified-Code Quality Comparison on Colab L4

This notebook runs the full 100M comparison only after the Goal 50 memorization gate passes. It compares Dense, MoP Full, Warm Adapter/Norm/Head 128, Cached Adapter/Norm/Head 128, and Cached Tail-Only LoRA Rank 8 on one balanced 10,000-lesson verified code-repair dataset.

Every enabled profile uses the same fixed split, tokenizer, sequence length, batch policy, optimizer-update budget, full held-out loss evaluation, and stratified generated-code evaluation from its best checkpoint. No quantization is used. Checkpoints, optimizer state, and activation caches stay outside the lightweight report.
"""
)
cells[1]["source"] = lines(
    """## 0. Settings

First run `colab_l4_goal50_100m_learning_gate.ipynb`. Keep its passing `learning_gate.json` available in the same Colab session or upload it when prompted below.

The shared `TARGET_EVAL_LOSS=0.85` is predeclared from the Goal 49 Dense best eval loss (`0.8022`) with a small relaxation. Every profile receives it before training, so baseline and sparse time-to-target remain comparable.
"""
)
cells[2]["source"] = lines(
    """REPO_URL = "https://github.com/NikanBHMNJ/MoP.git"
REPO_DIR = "/content/MoP"
REPORT_ID = "goal50_100m_quality_comparison"
REPORT_DIR = f"reports/{REPORT_ID}"
GATE_REPORT_PATH = "reports/goal50_100m_learning_gate/learning_gate.json"
REQUIRE_MEMORIZATION_GATE = True

OPTIMIZER_UPDATES = 2000
GRADIENT_ACCUMULATION_STEPS = 8
MAX_STEPS = OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION_STEPS
EVAL_EVERY_OPTIMIZER_UPDATES = 100
EVAL_EVERY_STEPS = EVAL_EVERY_OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION_STEPS
SAVE_EVERY_OPTIMIZER_UPDATES = 500
SAVE_EVERY_STEPS = SAVE_EVERY_OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION_STEPS
MAX_TRAIN_EXAMPLES = 10000
MAX_EVAL_EXAMPLES = 1000
COUNT_PER_CATEGORY = 2000
SPLIT_SEED = 42
TRAIN_SHUFFLE_SEED = 42
QUALITY_FORMAT = "fixed_code_xml"
ADAPTER_BOTTLENECK = 128

# Predeclared before all runs from Goal 49 Dense best eval loss (0.8022).
TARGET_EVAL_LOSS = 0.85
TARGET_EVAL_LOSS_SOURCE = "predeclared_goal49_dense_best_eval_0.8022_relaxed_to_0.85"
TARGET_EVAL_LOSS_WAS_CONFIGURED = TARGET_EVAL_LOSS is not None

CACHE_TEACHER_TOP_K = 16
CACHE_RECORDS_PER_SHARD = 256
DISTILLATION_WEIGHT = 0.2
DISTILLATION_TEMPERATURE = 2.0
HARD_EXAMPLE_REPLAY = True
HARD_EXAMPLE_REPLAY_LOSS_THRESHOLD = 1.5
HARD_EXAMPLE_REPLAY_MULTIPLIER = 2

# A later-run subset is allowed, but it is deterministically balanced across all five bug types.
GENERATION_EVAL_EXAMPLES = 250
GENERATION_MAX_NEW_TOKENS = 256

RUN_DENSE = True
RUN_MOP_FULL = True
RUN_WARM_ADAPTER_128 = True
RUN_CACHED_ADAPTER_128 = True
RUN_CACHED_LORA_8 = True
RUN_CACHED_LORA_16 = False
TEACHER_LABEL = "mop_full"
REQUIRE_CUDA = True
"""
)

setup_source = "".join(cells[5]["source"])
setup_source = setup_source.replace(
    '    sh("git pull --ff-only", check=False)',
    '    sh(f"git -C {repo_dir} pull --ff-only", check=False)',
)
setup_source += """

gate_path = Path(GATE_REPORT_PATH)
if REQUIRE_MEMORIZATION_GATE and not gate_path.exists():
    try:
        from google.colab import files

        print("Upload learning_gate.json from the Goal 50 memorization notebook.")
        uploaded = files.upload()
        if "learning_gate.json" in uploaded:
            gate_path = Path("learning_gate.json")
    except ImportError:
        pass
if REQUIRE_MEMORIZATION_GATE:
    if not gate_path.exists():
        raise FileNotFoundError("A passing Goal 50 learning_gate.json is required.")
    memorization_gate = read_json(gate_path)
    if memorization_gate.get("passed") is not True:
        raise RuntimeError("The 100M memorization gate did not pass. Do not run this comparison.")
    print("Memorization gate: PASS", gate_path)

if TARGET_EVAL_LOSS is None:
    raise ValueError(
        "Keep TARGET_EVAL_LOSS configured before training any comparison profile."
    )
"""
cells[5]["source"] = lines(setup_source)

cells[6]["source"] = lines("## 2. Build One Balanced 10,000-Lesson Verified Split\n")
cells[7]["source"] = lines(
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
)

config_source = "".join(cells[9]["source"])
config_source = config_source.replace(
    'CONFIG_DIR = Path("configs/jobs/colab_l4_goal49_quality")',
    'CONFIG_DIR = Path("configs/jobs/colab_l4_goal50_100m_quality")',
)
config_source = config_source.replace(
    '"max_steps": int(MAX_STEPS),\n',
    '"max_steps": int(MAX_STEPS),\n'
    '    "gradient_accumulation_steps": int(GRADIENT_ACCUMULATION_STEPS),\n'
    '    "eval_full_dataset": True,\n'
    '    "shuffle_train": True,\n'
    '    "train_shuffle_seed": int(TRAIN_SHUFFLE_SEED),\n',
)
config_source = config_source.replace(
    '"generation_max_new_tokens": int(GENERATION_MAX_NEW_TOKENS),\n',
    '"generation_max_new_tokens": int(GENERATION_MAX_NEW_TOKENS),\n'
    '    "generation_eval_use_best_checkpoint": True,\n'
    '    "generation_eval_include_train": False,\n'
    '    "generation_eval_stratify_by": "bug_type",\n',
)
config_source = config_source.replace(
    '            "split_seed": SPLIT_SEED,\n',
    '            "split_seed": SPLIT_SEED,\n'
    '            "train_shuffle_seed": TRAIN_SHUFFLE_SEED,\n'
    '            "optimizer_updates_requested": OPTIMIZER_UPDATES,\n'
    '            "microsteps_requested": MAX_STEPS,\n'
    '            "eval_full_dataset": True,\n'
    '            "target_eval_loss_source": TARGET_EVAL_LOSS_SOURCE,\n',
)
config_source = config_source.replace(
    '"dense", "configs/jobs/100m_dense_extended_efficiency.json", include_target=False',
    '"dense", "configs/jobs/100m_dense_extended_efficiency.json", include_target=True',
)
config_source = config_source.replace(
    '    include_target=False,\n)',
    '    include_target=True,\n)',
    1,
)
cells[9]["source"] = lines(config_source)

cells[10]["source"] = lines("## 4. Train Dense, MoP Full, And Warm Adapter 128\n")
cells[11]["source"] = lines(
    """if RUN_DENSE:
    train_config("dense", dense_config)
if RUN_MOP_FULL:
    train_config("mop_full", mop_full_config)

if RUN_WARM_ADAPTER_128:
    if "mop_full" not in RUNS:
        raise RuntimeError("Warm Adapter 128 requires RUN_MOP_FULL=True.")
    warm_config = make_config(
        "warm_adapter_norm_head_128",
        "configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json",
        {
            "fast_adapter_bottleneck_dim": ADAPTER_BOTTLENECK,
            "resume_from_checkpoint": RUNS["mop_full"]["checkpoint_path"],
            "base_checkpoint_path": RUNS["mop_full"]["checkpoint_path"],
            "resume_model_only": True,
            "save_trainable_only_checkpoints": True,
        },
    )
    train_config("warm_adapter_norm_head_128", warm_config)
"""
)

report_source = "".join(cells[15]["source"])
report_source = report_source.replace(
    '    "generation_eval.json",\n',
    '    "generation_eval.json",\n    "ground_truth_controls.json",\n',
)
report_source = report_source.replace(
    '        "target_was_auto_derived": not TARGET_EVAL_LOSS_WAS_CONFIGURED,\n',
    '        "target_was_auto_derived": False,\n'
    '        "target_eval_loss_source": TARGET_EVAL_LOSS_SOURCE,\n'
    '        "optimizer_updates": OPTIMIZER_UPDATES,\n'
    '        "microsteps": MAX_STEPS,\n'
    '        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,\n'
    '        "eval_full_dataset": True,\n'
    '        "split_stratify_by": "bug_type",\n'
    '        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,\n'
    '        "generation_eval_stratify_by": "bug_type",\n',
)
cells[15]["source"] = lines(report_source)

summary_source = "".join(cells[16]["source"])
gate_code = '''evidence = {}
row_by_label = {
    label_by_run_id.get(row.get("run_id"), row.get("run_id", "")): row for row in rows
}
for label, record in RUNS.items():
    run_dir = Path("gpu_runs") / record["run_id"]
    metrics = read_json(run_dir / "metrics.json")
    generation = read_json(run_dir / "generation_eval.json")
    controls = read_json(run_dir / "ground_truth_controls.json")
    summary = generation["splits"]["eval"]["summary"]
    evidence[label] = {
        "ground_truth_controls_passed": controls["passed"],
        "generation_checkpoint": generation["checkpoint"],
        "category_count": len(summary.get("per_category", {})),
        "fixed_code_complete_rate": summary["gen_fixed_code_complete_rate"],
        "syntax_pass_rate": summary["gen_syntax_pass_rate"],
        "verifier_pass_rate": summary["gen_verifier_pass_rate"],
        "exact_match_rate": summary["gen_exact_match_rate"],
        "latest_eval_examples": metrics.get("latest_eval_examples"),
        "full_eval_examples": metrics.get("data", {}).get("eval_examples"),
        "optimizer_updates": metrics.get("optimizer_steps"),
        "sequence_length_statistics": metrics.get("data", {}).get("sequence_length_statistics"),
    }

dense_row = row_by_label.get("dense", {})
cached_labels = [label for label in RUNS if label.startswith("cached_")]


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def untruncated(item):
    stats = item.get("sequence_length_statistics") or {}
    return all(
        (stats.get(split_name) or {}).get("truncated_examples") == 0
        for split_name in ("train", "eval")
    )


def generation_budget_covers_targets(item):
    stats = item.get("sequence_length_statistics") or {}
    maxima = [
        (stats.get(split_name) or {}).get("max_target_tokens")
        for split_name in ("train", "eval")
    ]
    return all(numeric(value) and value <= GENERATION_MAX_NEW_TOKENS for value in maxima)


checks = {
    "all_ground_truth_controls": all(item["ground_truth_controls_passed"] for item in evidence.values()),
    "all_generation_from_best_checkpoint": all(item["generation_checkpoint"]["source"] == "best_eval" for item in evidence.values()),
    "all_five_categories": all(item["category_count"] == 5 for item in evidence.values()),
    "full_held_out_loss": all(item["latest_eval_examples"] == item["full_eval_examples"] for item in evidence.values()),
    "optimizer_update_budget": all(item["optimizer_updates"] == OPTIMIZER_UPDATES for item in evidence.values()),
    "no_train_or_eval_truncation": all(untruncated(item) for item in evidence.values()),
    "generation_budget_covers_targets": all(generation_budget_covers_targets(item) for item in evidence.values()),
    "cached_complete_framing_at_least_90pct": all(evidence[label]["fixed_code_complete_rate"] >= 0.90 for label in cached_labels),
    "cached_syntax_at_least_80pct": all(evidence[label]["syntax_pass_rate"] >= 0.80 for label in cached_labels),
    "cached_verifier_at_least_20pct": all(evidence[label]["verifier_pass_rate"] >= 0.20 for label in cached_labels),
    "cached_exact_match_nonzero": all(evidence[label]["exact_match_rate"] > 0.0 for label in cached_labels),
    "cached_peak_reserved_below_dense": numeric(dense_row.get("peak_reserved_gb")) and all(numeric(row_by_label.get(label, {}).get("peak_reserved_gb")) and row_by_label[label]["peak_reserved_gb"] < dense_row["peak_reserved_gb"] for label in cached_labels),
    "cached_throughput_at_least_80pct_dense": numeric(dense_row.get("tokens_per_sec")) and all(numeric(row_by_label.get(label, {}).get("tokens_per_sec")) and row_by_label[label]["tokens_per_sec"] >= 0.8 * dense_row["tokens_per_sec"] for label in cached_labels),
}
acceptance = {
    "passed": bool(cached_labels) and all(checks.values()),
    "checks": checks,
    "thresholds": {
        "fixed_code_complete_rate": 0.90,
        "syntax_pass_rate": 0.80,
        "verifier_pass_rate": 0.20,
        "exact_match_rate": "greater_than_zero",
        "cached_tokens_per_sec_vs_dense": 0.80,
    },
    "evidence": evidence,
}
write_json(REPORT_PATH / "acceptance_gates.json", acceptance)

'''
summary_source = summary_source.replace("readme = f\"\"\"# Goal 49", gate_code + 'readme = f"""# Goal 50')
summary_source = summary_source.replace(
    "Generated by `notebooks/colab_l4_goal49_verified_code_quality_report.ipynb`.",
    "Generated by `notebooks/colab_l4_goal50_100m_quality_comparison.ipynb`.",
)
summary_source = summary_source.replace(
    "- Quantization: none\n",
    "- Quantization: none\n- Target source: `{TARGET_EVAL_LOSS_SOURCE}`\n- Optimizer updates per profile: `{OPTIMIZER_UPDATES}`\n- Full held-out loss evaluation: `True`\n- Generation subset: `{GENERATION_EVAL_EXAMPLES}`, stratified across five bug types\n- Acceptance gates: `{'PASS' if acceptance['passed'] else 'FAIL'}`\n",
)
summary_source = summary_source.replace(
    "Each `runs/<profile>/generation_eval.json` contains generated samples and verifier outcomes.",
    "Each `runs/<profile>/generation_eval.json` records the exact best checkpoint, stratified generated samples, complete XML, syntax, exact-match, verifier, and per-category outcomes. `ground_truth_controls.json` must remain 100% passing.",
)
summary_source = summary_source.replace(
    "Treat a cached profile as a quality-preserving efficiency win only when loss and generated-code quality remain close to or better than the baselines while a named VRAM, throughput, trainable-ratio, or checkpoint-size axis improves.",
    "`acceptance_gates.json` is the claim boundary. Do not scale to 1B or claim quality-preserving efficiency unless it passes; inspect the recorded per-category failures when it does not.",
)
cells[16]["source"] = lines(summary_source)

cells[17]["source"] = lines(
    """## 7. Download The Report

The ZIP contains only lightweight comparison evidence, acceptance gates, generated samples, controls, and metadata. It excludes model weights, optimizer state, and activation caches.
"""
)

notebook["metadata"]["colab"]["name"] = OUTPUT.name
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
