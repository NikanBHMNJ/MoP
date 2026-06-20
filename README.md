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
- Native non-reentrant PyTorch activation checkpointing for dense, shared, and
  routed transformer blocks.
- Routed FFN expert blocks with top-k example or token routing.
- Dense-to-routed warm starts that clone dense FFN weights into routed experts.
- Module-routed low-rank deltas for attention Q/K/V, attention output, and FFN
  up/down projections.
- Fixed-split coding dataset preparation for fair dense-vs-sparse comparisons.
- Generated-code evaluation metrics, including exact match and verifier pass
  rate.
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

## Current Research Direction

Goal 47 implemented the infrastructure needed to reduce sparse-run loss without
giving up the efficiency story:

- Warm-start sparse runs from a learned dense or full-MoP checkpoint instead of
  training adapters on a random frozen base.
- Train adapters with optional norm/head updates for a small capacity increase.
- Cache frozen-prefix activations for repeated sparse-tail sweeps.
- Save trainable-only checkpoints so artifact size matches the sparse claim.
- Use routed FFN experts and internal low-rank deltas as quality recovery paths.
- Gate claims with eval loss, throughput, VRAM, checkpoint size, generated-code
  exact match, and verifier pass rate.

This work is implemented and tested. The first L4 warm sparse report is
available under `reports/v0_46_0_l4_warm_sparse_comparison/`, but broader claims
still need longer runs, repeated seeds, and task-specific quality checks.

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
mopforge gpu write-warm-sparse-sweep \
  --base-checkpoint <mop_full_run_id_or_checkpoint> \
  --dataset-ref <dataset_id@version_id> \
  --dataset-split-id <split_id> \
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
axis.

## Documentation

- [Docs index](docs/README.md)
- [GPU quickstart](docs/gpu_quickstart.md)
- [Colab L4 TinyStories v0.46.0 efficiency comparison notebook](notebooks/colab_l4_v046_efficiency_comparison.ipynb)
- [GPU efficiency benchmarking](docs/gpu_efficiency_benchmarking.md)
- [Warm sparse comparison template](docs/warm_sparse_efficiency_comparison_template.md)
- [Goal 46 GPU efficiency report](reports/goal46_gpu_efficiency/README.md)
- [v0.46.0 L4 warm sparse comparison report](reports/v0_46_0_l4_warm_sparse_comparison/README.md)
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
