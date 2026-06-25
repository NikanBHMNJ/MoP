# GPU Efficiency Benchmarking

MoP-Forge reports efficiency only when the measured axis and comparison scope
are explicit. A sparse run is not an efficiency win by default; it must preserve
the predeclared quality gate while improving a named resource axis.

## Current Report

The current measured evidence is:

```text
reports/verified_code_repair_100m_l4/
```

It compares Dense, MoP Full, Warm Adapter/Norm/Head 128, Cached
Adapter/Norm/Head 128, and Cached Tail-Only LoRA Rank 8 on the same fixed
verified code-repair split.

## Required Axes

Every report should include:

- train loss,
- eval loss,
- best eval loss,
- time-to-target loss,
- tokens-to-target loss,
- tokens/sec,
- samples/sec,
- peak allocated VRAM,
- peak reserved VRAM,
- final reserved VRAM,
- trainable parameter ratio,
- active parameter ratio when routed execution is used,
- checkpoint size,
- generated-code syntax pass,
- exact match,
- verifier or task pass rate,
- hardware, precision, driver, CUDA, and PyTorch metadata.

## Compare Runs

```bash
mopforge gpu compare-runs <dense_run_id> <sparse_run_id> \
  --output outputs/gpu_efficiency_comparison.json \
  --output-csv outputs/gpu_efficiency_comparison.csv
```

## Gate Sparse Efficiency

```bash
mopforge gpu gate-efficiency \
  --dense-run <dense_run_id> \
  --sparse-run <sparse_run_id> \
  --output outputs/gpu_efficiency_gate_report.json
```

The gate checks same-split quality, generated-code quality when present,
throughput, VRAM, sparse checkpoint policy, quantization status, and reported
efficiency axes.

## Claim Card

After the GPU gate, create a public claim card:

```bash
mopforge claim scaffold --report-dir reports/<report-id> \
  --claim-statement "<measured, scoped efficiency claim>" \
  --academic-level A2 \
  --product-level P2 \
  --output outputs/<report-id>-claim-card.json

mopforge claim validate outputs/<report-id>-claim-card.json
```

## Interpretation

Use narrow wording:

```text
This report supports the measured quality and efficiency axes on the named
dataset split, hardware target, model size, seed, and training budget.
```

Do not claim broad model quality, hardware feasibility, production service
readiness, or customer value without separate evidence.
