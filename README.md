# MoP-Forge

**Version:** `0.46.0`<br>
**Status:** v1.0-beta research framework with single-GPU efficiency benchmarking.

MoP-Forge is a local-first research framework for testing **Mixture-of-Parameters (MoP)** training ideas. The current release supports CPU-safe smoke tests, local training artifacts, single-GPU research runs, GPU checkpointing, and now **GPU-efficiency comparison tooling** for dense vs MoP training modes.

This update adds the first serious evidence-oriented workflow:

```text
100M Dense
vs
100M MoP Full
vs
100M MoP Adapter-Only
```

The goal is no longer only “does it run?” The goal is now:

```text
Can MoP reduce trainable parameters, memory, checkpoint size, or training cost
while keeping loss reasonably close to a dense baseline?
```

---

## What Goal 46 Added

Goal 46 added GPU-efficiency benchmarking and sparse MoP training modes.

### GPU efficiency metrics

GPU training now writes nested efficiency metrics into:

```text
gpu_runs/<run_id>/metrics.json
gpu_runs/<run_id>/gpu_training_result.json
```

Tracked metrics include:

```text
tokens_per_sec
samples_per_sec
step_time_sec
total_train_time_sec
peak_allocated_gb
peak_reserved_gb
final_allocated_gb
final_reserved_gb
total_params
trainable_params
frozen_params
trainable_param_ratio
active_param_estimate
active_param_ratio
active_module_density
active_adapter_density
generated_condition_density
checkpoint_size_mb
```

CUDA memory tracking uses PyTorch peak/current allocated and reserved memory APIs when CUDA is available, and safely records `null` on CPU/no-CUDA runs.

### Sparse / parameter-efficient MoP policies

The framework now supports multiple trainable-policy modes for MoP experiments:

```text
all
adapters_only
modules_only
core_frozen
router_adapters_only
```

These policies allow comparisons between full dense training and parameter-efficient MoP variants.

### Colab-safe 100M configs

Goal 46 added validated Colab/L4-safe job configs:

```text
configs/jobs/100m_dense_colab_efficiency.json
configs/jobs/100m_mop_full_colab_efficiency.json
configs/jobs/100m_mop_adapters_only_colab_efficiency.json
configs/jobs/100m_mop_core_frozen_colab_efficiency.json
configs/jobs/100m_mop_router_adapters_colab_efficiency.json
```

### Comparison tooling

Goal 46 added:

```bash
mopforge gpu compare-runs <RUN_ID_1> <RUN_ID_2> ...
```

and:

```bash
python scripts/compare_gpu_runs.py
```

These tools compare dense and MoP GPU runs and export JSON/CSV comparison reports.

---

## First 100M Efficiency Result

A Colab/L4 run compared:

```text
100M Dense
100M MoP Full
100M MoP Adapter-Only
```

### Run IDs

```text
Dense:
20260617T094127Z-100m-dense-colab-efficiency-a4e4bd2e

MoP Full:
20260617T094141Z-100m-mop-full-colab-efficiency-539270ba

MoP Adapter-Only:
20260617T094157Z-100m-mop-adapters-only-colab-efficiency-47bab16e
```

### Result table

| Model | Train loss | Eval loss | Tokens/sec | Peak reserved VRAM | Trainable ratio | Active ratio | Checkpoint size | Device |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Dense | 3.0467 | 3.1705 | 11,286.54 | 1.9844 GB | 1.0 | 1.0 | 987.14 MB | cuda:0 |
| MoP Full | 3.0377 | 3.1691 | 10,402.15 | 2.1367 GB | 1.0 | 1.0 | 1078.05 MB | cuda:0 |
| MoP Adapter-Only | 5.1322 | 5.1653 | 26,812.31 | 0.4961 GB | 0.000842 | 1.0 | 365.86 MB | cuda:0 |

---

## Interpretation

This result is meaningful, but it must be interpreted carefully.

### What worked

```text
100M dense training works on GPU.
100M MoP full training works on GPU.
100M MoP adapter-only training works on GPU.
Efficiency metrics are recorded.
Comparison reports work.
CUDA + BF16 path works.
Checkpoint size, throughput, and VRAM are measurable.
```

### What the result shows

MoP Full reached almost the same eval loss as Dense:

```text
Dense eval loss:    3.1705
MoP Full eval loss: 3.1691
```

However, MoP Full is not more efficient yet because it trains all parameters and uses more memory/checkpoint space.

MoP Adapter-Only is much more efficient:

```text
~2.38x higher tokens/sec than dense
~75% lower peak reserved VRAM than dense
~63% smaller checkpoint than dense
~99.916% fewer trainable parameters than dense
```

But it has worse eval loss:

```text
Dense eval loss:            3.1705
MoP Adapter-Only eval loss: 5.1653
Loss gap:                  +1.9948
```

### Honest claim

This update proves:

```text
MoP-Forge can now measure GPU efficiency and run sparse MoP training modes.
```

It does **not** yet prove:

```text
MoP is better than dense models.
```

The current evidence says:

```text
MoP Full matches dense quality but is not more efficient.
MoP Adapter-Only is much more efficient but currently lower quality.
```

---

## How to Reproduce the 100M Efficiency Experiment

### 1. Install

```bash
pip install -e .[dev]
mopforge doctor
mopforge runtime detect
```

### 2. Prepare data

Generate coding/debugging lessons:

```bash
python examples/generate_coding_bugfix_lessons.py
```

Create or provide a corpus file expected by the configs:

```text
data/colab_tinystories_corpus.jsonl
```

### 3. Validate configs

```bash
mopforge gpu validate configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_full_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_adapters_only_colab_efficiency.json
```

### 4. Train

```bash
mopforge gpu train configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_full_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_adapters_only_colab_efficiency.json
```

### 5. Compare

```bash
mopforge gpu list
```

Then:

```bash
mopforge gpu compare-runs \
  <DENSE_RUN_ID> \
  <MOP_FULL_RUN_ID> \
  <MOP_ADAPTER_RUN_ID> \
  --gpu-runs-dir gpu_runs \
  --output outputs/100m_efficiency_comparison.json
```

Export CSV too:

```bash
python scripts/compare_gpu_runs.py \
  --runs <DENSE_RUN_ID> <MOP_FULL_RUN_ID> <MOP_ADAPTER_RUN_ID> \
  --gpu-runs-dir gpu_runs \
  --output-json outputs/100m_efficiency_comparison.json \
  --output-csv outputs/100m_efficiency_comparison.csv
```

---

## Recommended Next Experiment

The next experiment should not jump to 500M or 1B yet.

Run a longer 100M experiment:

```text
100M Dense — 500 steps
100M MoP Full — 500 steps
100M MoP Adapter-Only — 500 steps
100M MoP Core-Frozen — 500 steps
```

The key research question is:

```text
Can adapter-only or core-frozen MoP close the loss gap while keeping
its trainable-parameter, VRAM, and checkpoint-size advantages?
```

A promising future result would look like:

| Model | Eval loss | Trainable params | VRAM | Verdict |
|---|---:|---:|---:|---|
| Dense | strong baseline | high | high | quality baseline |
| MoP Full | similar | high | high | architecture check |
| MoP Adapter-Only | slightly worse | tiny | low | efficiency candidate |
| MoP Core-Frozen | close to dense | much lower | lower/similar | strongest candidate |

---

## Current Limitations

- This is a short Colab/L4 experiment, not a paper-grade result.
- Adapter-only MoP is efficient but not yet quality-competitive.
- Active parameter ratio is still `1.0` in the current adapter-only run, so future work should improve active routing sparsity.
- No FSDP, DeepSpeed, tensor parallelism, custom CUDA kernels, or production distributed training.
- MoP routing and Fast Parameters remain PyTorch-level experimental paths.
- The result does not prove large-scale MoP superiority.
- More steps, more seeds, and larger experiments are needed before making research claims.

---

## Summary

Goal 46 moves MoP-Forge from “GPU-compatible” to “GPU-efficiency measurable.”

The framework can now run dense vs MoP efficiency experiments and produce concrete metrics for:

```text
loss
throughput
VRAM
trainable parameters
active parameters
checkpoint size
routing/adapters/generated density
```

The first result is encouraging for framework maturity:

```text
MoP Adapter-Only is much faster and lighter,
but needs more work to close the quality gap.
```

This is a successful research-framework milestone and a clear starting point for the next round of MoP efficiency experiments.
