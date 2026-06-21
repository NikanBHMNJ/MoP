# MoP-Forge

**Version:** `0.46.0`
**Status:** local-first research framework with single-GPU efficiency tooling

MoP-Forge is a research codebase for testing **Mixture-of-Parameters (MoP)**
training ideas against dense baselines. It focuses on evidence: every useful
efficiency claim should name the axis being improved, such as trainable
parameters, VRAM, checkpoint size, active compute, throughput, or generated-code
quality.

MoP-Forge is not a production distributed LLM training framework. It does not include
FSDP, DeepSpeed, custom CUDA kernels, model downloads, cloud launchers, or a
hardened multi-GPU training stack.

## What Is Implemented

MoP-Forge currently includes:

- CPU-safe smoke training and test coverage for local development.
- Single-device GPU training profiles with CUDA/BF16 support when available.
- Dense, full-MoP, adapter-only, core-frozen, routed-FFN, and warm sparse
  experiment profiles.
- Trainable-parameter policies for sparse fine-tuning:
  `all`, `adapters_only`, `adapters_norm_head`, `modules_only`,
  `core_frozen`, `router_only`, and `router_adapters_only`.
- Model-only checkpoint resume for warm-started sparse runs.
- Trainable-only sparse checkpoints with base-checkpoint references.
- Frozen-prefix execution and activation-cache training for sparse tails.
- Cached teacher top-k distillation for sparse-tail code training.
- Cached sparse training can offload unused frozen backbone modules from CUDA
  before the trainable tail phase.
- Cache-compatible tail-only LoRA keeps low-rank deltas after the hidden-state
  boundary, so LoRA rank 8/16 students can train without restoring the frozen
  backbone to CUDA.
- Opt-in verified fixed-code target framing for small code-quality students via
  `--quality-format fixed_code_xml`.
- Native non-reentrant PyTorch activation checkpointing for dense, shared, and
  routed transformer blocks.
- Routed FFN expert blocks with top-k example or token routing.
- Dense-to-routed warm starts that clone dense FFN weights into routed experts.
- Module-routed low-rank deltas for attention Q/K/V, attention output, and FFN
  up/down projections.
- Fixed-split coding dataset preparation for fair dense-vs-sparse comparisons.
- Optional `bug_type`-stratified fixed splits and deterministic train-loader
  reshuffling at real epoch boundaries, with the seed and epoch counters saved
  in run metadata.
- Full held-out loss evaluation for quality runs, including the exact number of
  batches and examples used for every evaluation.
- Generated-code evaluation metrics, including exact match and verifier pass
  rate, with extraction support for `<fixed_code>...</fixed_code>` outputs.
  Quality evaluation can restore the best eval-loss checkpoint, balance samples
  by bug category, evaluate train and held-out splits separately, and save the
  checkpoint path, complete-XML rate, per-category failures, and generated
  samples to `generation_eval.json`.
- Automatic raw/XML ground-truth verifier controls and pre-truncation
  prompt/target/sequence-length statistics for code-quality reports.
- JSON/CSV GPU run comparison and sparse-efficiency acceptance gates.

## Latest Evidence

The first committed GPU efficiency evidence is the Goal 46 100M Colab/L4
comparison:

`reports/goal46_gpu_efficiency/`

It compares:

- 100M Dense
- 100M MoP Full
- 100M MoP Adapter-Only

| Model | Train loss | Eval loss | Tokens/sec | Peak reserved VRAM | Trainable ratio | Active ratio | Checkpoint size | Device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense | 3.0467 | 3.1705 | 11,286.54 | 1.9844 GB | 1.0 | 1.0 | 987.14 MB | cuda:0 |
| MoP Full | 3.0377 | 3.1691 | 10,402.15 | 2.1367 GB | 1.0 | 1.0 | 1078.05 MB | cuda:0 |
| MoP Adapter-Only | 5.1322 | 5.1653 | 26,812.31 | 0.4961 GB | 0.000842 | 1.0 | 365.86 MB | cuda:0 |

Honest interpretation:

- MoP Full matched dense quality in this run, but was not more efficient.
- MoP Adapter-Only was faster, used less VRAM, used far fewer trainable
  parameters, and wrote a smaller checkpoint, but its eval loss was worse.
- This proves that MoP-Forge can measure GPU efficiency and run sparse MoP
  modes. It does not prove that MoP is better than dense.

Useful derived points:

- MoP Adapter-Only was about `2.38x` faster than Dense by tokens/sec.
- MoP Adapter-Only used about `75%` less peak reserved VRAM than Dense.
- MoP Adapter-Only used about `99.916%` fewer trainable parameters than Dense.
- MoP Adapter-Only checkpoint size was about `63%` smaller than Dense.

A newer v0.46.0 Colab/L4 TinyStories warm sparse comparison is available under:

`reports/v0_46_0_l4_warm_sparse_comparison/`

It compares Dense, MoP Full, Warm Adapter Norm/Head 64, and Warm LoRA Rank 8
for 300 steps on the same 6,000-record TinyStories corpus slice. In this short
run, the two warm sparse profiles had lower eval loss than Dense, higher
tokens/sec, lower reserved VRAM, far fewer trainable parameters, and much
smaller trainable-only checkpoints. Treat this as promising workflow evidence,
not a paper-grade conclusion.

The first Goal 48 code-dataset cached sparse L4 report is available under:

`reports/goal48_code_cached_sparse_efficiency/`

It compares Dense, MoP Full, Warm Adapter Norm/Head 64, and Cached Warm Adapter
Norm/Head 64 with teacher top-k KL on the same fixed code split. The cached
sparse run was much faster, used far less peak reserved VRAM, offloaded the
frozen backbone, and kept a tiny trainable-only checkpoint. Its generated-code
quality is not proven in this report, and `target_eval_loss` was not configured,
so treat it as evidence that the cached sparse efficiency path works rather
than proof of same-quality sparse superiority.

The Goal 49 verified code-quality L4 report is available under:

`reports/goal49_verified_code_quality_efficiency/`

It uses one fixed `fixed_code_xml` code-repair split and compares Dense, MoP
Full, Warm Adapter/Norm/Head 128, Cached Adapter/Norm/Head 128, and Cached
Tail-Only LoRA Rank 8. No quantization was used.

| Profile | Best eval loss | Tokens/sec | Peak reserved VRAM | Trainable ratio | Checkpoint | Syntax pass | Verifier pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.8022 | 9,346.82 | 1.8652 GB | 1.0000 | 987.15 MB | 25.00% | 0.00% |
| MoP Full | 0.9046 | 8,603.26 | 1.9707 GB | 1.0000 | 1,055.61 MB | 34.38% | 0.00% |
| Warm Adapter/Norm/Head 128 | 0.6206 | 40,153.43 | 1.5723 GB | 0.0085 | 7.66 MB | 0.00% | 0.00% |
| Cached Adapter/Norm/Head 128 | 0.4685 | 87,518.63 | 0.0605 GB | 0.0085 | 7.66 MB | 50.00% | 0.00% |
| Cached Tail-Only LoRA Rank 8 | 0.4980 | 71,133.54 | 0.0820 GB | 0.0089 | 8.05 MB | 50.00% | 0.00% |

Compared with Dense, Cached Adapter/Norm/Head 128 measured about `9.36x`
higher throughput, `30.83x` lower peak reserved VRAM, `37.23x` lower peak
allocated VRAM, and a `128.79x` smaller checkpoint. Its best eval loss was also
lower. Cached Tail-Only LoRA Rank 8 retained similar gains.

This is a strong cached-training efficiency result, but not an output-quality
win yet. Across 32 generated examples per profile, both cached students reached
`50%` syntax pass while exact match and verifier pass remained `0%` for every
profile. The target loss was derived after the baseline runs, so baseline
time-to-target values are unavailable and must not be inferred.

The Goal 50 100M memorization-gate report is available under:

`reports/goal50_100m_learning_gate/`

The protocol checks worked: all five categories were represented, full eval was
used, raw/XML ground-truth controls passed, no examples were truncated, 1,000
optimizer updates completed, and generation restored the best checkpoint.
The corrected rerun passed: train and held-out XML completion, syntax pass,
verifier pass, and exact match were all `100%`, with best eval loss reaching
`0.0000904`. Phase C is now unblocked; the 1B run still waits for the full 100M
comparison.

The initial failed run exposed a concrete boundary mismatch: training encoded
`BOS + prompt + target`, while greedy generation encoded
`BOS + prompt + EOS` before predicting the target. Greedy generation now uses
the same BOS-only prompt boundary as training. The passing rerun confirms that
this mismatch, rather than model capacity or the verifier, caused the failure.

## Current Research Direction

Goal 50 first tests whether the 100M pipeline can learn its narrow verified
repair contract before spending an L4 session on a 1B model. The protocol now:

- separates microsteps from optimizer updates,
- reshuffles the train set deterministically at every real epoch,
- evaluates held-out loss over the complete eval split,
- generates from the saved best eval-loss checkpoint,
- balances diagnostic generation across all five bug categories,
- reports complete XML, syntax, verifier, exact-match, and failure metrics per
  category,
- verifies raw and XML ground-truth controls,
- records truncation statistics and the generation budget.

The cached sparse direction remains in place:

- Warm-start sparse runs from a learned dense or full-MoP checkpoint instead of
  training adapters on a random frozen base.
- Train adapters with optional norm/head updates for a small capacity increase.
- Cache frozen-prefix activations for repeated sparse-tail sweeps.
- Cache teacher top-k logits and add a KL distillation term during sparse-tail
  training, so code runs can learn from a Dense or MoP Full teacher without
  keeping that teacher resident during the sparse phase.
- Optionally replay high-loss cached examples using per-record teacher CE loss.
- Offload frozen backbone modules from CUDA for cached sparse-tail training.
- Save the best eval-loss checkpoint and record time/tokens-to-target-loss when
  a shared `target_eval_loss` is configured.
- Save trainable-only checkpoints so artifact size matches the sparse claim.
- Use routed FFN experts and internal low-rank deltas as quality recovery paths.
- Gate claims with eval loss, throughput, VRAM, checkpoint size, generated-code
  exact match, and verifier pass rate.
- For the next output-quality run, frame code-repair targets as verified
  `<fixed_code>...</fixed_code>` blocks so small students learn a narrow
  repair/completion contract instead of broad free-form code generation.
- Compare Adapter/Norm/Head 128 with cache-compatible tail-only LoRA rank 8/16,
  and evaluate every profile on generated samples after training.

This work is implemented and tested. The first L4 warm sparse report is
available under `reports/v0_46_0_l4_warm_sparse_comparison/`, but broader claims
still need longer runs, repeated seeds, and task-specific quality checks.

The prior full quality comparison is
`notebooks/colab_l4_goal49_verified_code_quality_report.ipynb`. It builds a
downloadable lightweight report with comparison JSON/CSV, run metadata, and
generated-code samples while excluding caches and model weights.

The completed diagnostic is
`notebooks/colab_l4_goal50_100m_learning_gate.ipynb`. It trains the balanced
50-lesson 100M Dense memorization test for 1,000 optimizer updates, evaluates
full train and held-out generation from the best checkpoint, and emits an
explicit pass/fail report. The first measured run failed that gate, so the 1B
run remained blocked. The corrected rerun passed every gate, so the full 100M
comparison may proceed; 1B remains gated on that comparison.

After that gate passes, run
`notebooks/colab_l4_goal50_100m_quality_comparison.ipynb`. It compares the five
100M Dense, full-MoP, warm sparse, cached adapter, and cached tail-LoRA profiles
on 10,000 balanced verified lessons for 2,000 optimizer updates each. It writes
`acceptance_gates.json` and refuses to start without a passing memorization gate
and a preconfigured shared target eval loss. The committed target is `0.85`,
declared before all runs from the Goal 49 Dense `0.8022` baseline evidence.

## Quickstart

Install in editable mode:

```bash
pip install -e .[dev]
mopforge doctor
mopforge runtime detect
```

Run the CPU-safe GPU trainer smoke path:

```bash
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
```

On a CUDA machine:

```bash
mopforge gpu train configs/jobs/tiny_gpu_smoke.json --device cuda --precision bf16
```

## Reproducing The Goal 46 Evidence

Validate the 100M profiles:

```bash
mopforge gpu validate configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_full_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_adapters_only_colab_efficiency.json
```

Train and compare:

```bash
mopforge gpu train configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_full_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_adapters_only_colab_efficiency.json

mopforge gpu compare-runs <dense_run_id> <mop_full_run_id> <adapter_run_id> \
  --output outputs/100m_efficiency_comparison.json \
  --output-csv outputs/100m_efficiency_comparison.csv
```

## Recommended Next Experiment

Use the fixed-split dataset and extended 100M profiles:

```bash
mopforge gpu prepare-efficiency-data --count-per-category 100 --split-seed 42
mopforge gpu train configs/jobs/100m_dense_extended_efficiency.json
mopforge gpu train configs/jobs/100m_mop_full_extended_efficiency.json
```

For a small-model code-quality run, prepare the same fixed split with verified
fixed-code targets:

```bash
mopforge gpu prepare-efficiency-data \
  --count-per-category 10 \
  --split-seed 42 \
  --stratify-by bug_type \
  --quality-format fixed_code_xml
```

Build the cached teacher signal once from the warm teacher:

```bash
mopforge gpu cache-activations configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json \
  --checkpoint <mop_full_run_id_or_checkpoint> \
  --output outputs/warm_sparse_cache_manifest.json \
  --teacher-top-k 16 \
  --records-per-shard 512
```

Write cached sparse distillation profiles:

```bash
mopforge gpu write-warm-sparse-sweep \
  --base-checkpoint <mop_full_run_id_or_checkpoint> \
  --dataset-ref <dataset_id@version_id> \
  --dataset-split-id <split_id> \
  --activation-cache-path outputs/warm_sparse_cache_manifest.json \
  --cached-distillation-weight 0.2 \
  --cached-distillation-temperature 2.0 \
  --cached-distillation-top-k 16 \
  --hard-example-replay \
  --hard-example-replay-loss-threshold <teacher_ce_loss_threshold> \
  --hard-example-replay-multiplier 2 \
  --target-eval-loss <dense_or_mop_full_target_loss> \
  --output-dir configs/jobs/warm_sparse_sweep
```

Then run the warm sparse profiles and gate the claim:

```bash
mopforge gpu gate-efficiency \
  --dense-run <dense_run_id> \
  --sparse-run <sparse_run_id> \
  --output outputs/warm_sparse_gate_report.json
```

Do not claim same-quality sparse efficiency unless the sparse run remains close
to Dense eval loss and generated-code quality while improving a named efficiency
axis. Use `target_eval_loss` only with the same fixed split and eval cadence, so
time-to-target-loss is comparable across Dense, MoP Full, warm sparse, and
cached sparse runs.

## Documentation

- [Docs index](docs/README.md)
- [GPU quickstart](docs/gpu_quickstart.md)
- [Colab L4 TinyStories v0.46.0 efficiency comparison notebook](notebooks/colab_l4_v046_efficiency_comparison.ipynb)
- [Colab L4 Goal 48 code cached-sparse report notebook](notebooks/colab_l4_goal48_code_cached_sparse_report.ipynb)
- [Colab L4 Goal 49 verified-code quality report notebook](notebooks/colab_l4_goal49_verified_code_quality_report.ipynb)
- [GPU efficiency benchmarking](docs/gpu_efficiency_benchmarking.md)
- [Warm sparse comparison template](docs/warm_sparse_efficiency_comparison_template.md)
- [Goal 46 GPU efficiency report](reports/goal46_gpu_efficiency/README.md)
- [v0.46.0 L4 warm sparse comparison report](reports/v0_46_0_l4_warm_sparse_comparison/README.md)
- [Goal 48 code cached-sparse L4 report](reports/goal48_code_cached_sparse_efficiency/README.md)
- [Goal 49 verified code-quality L4 report](reports/goal49_verified_code_quality_efficiency/README.md)
- [Known limitations](docs/known_limitations.md)

## Validation

The current implementation was validated with:

```text
python -m pytest -q
python scripts/release_check.py --quick-examples
```

Latest local result before this README update:

```text
414 passed, 1 skipped
release checks passed for version 0.46.0
```

## Limitations

- The latest warm sparse and routed-expert features are implemented, but their
  claimed GPU benefit must be established by new CUDA runs.
- The Goal 46 report is a short 100M Colab/L4 experiment, not a paper-grade
  result.
- Adapter-only MoP is efficient in the current evidence, but not yet
  quality-competitive.
- Active parameter and FLOP estimates are model-level approximations, not custom
  kernel measurements.
- CPU fallback validates functionality, not GPU performance.
- Generated-code verification is local and intentionally lightweight.

## Project Position

MoP-Forge is now a measurement-oriented MoP research framework. It can run
dense and sparse experiments, preserve lightweight evidence artifacts, and make
claims testable. The v1.0-beta path is not another implementation-only claim;
it is a longer, repeated GPU comparison showing whether warm sparse MoP can
close the loss gap while keeping a measurable efficiency advantage.
