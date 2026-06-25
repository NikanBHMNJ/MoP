# GPU Quickstart

MoP-Forge provides single-device CUDA/BF16 training, cached sparse-tail
training, and torchrun DDP/FSDP admission paths. CPU tests validate framework
correctness; performance and feasibility claims require measured GPU reports.

## Local Validation

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/tiny_gpu_smoke.json
```

Run the tiny profile on CUDA when available:

```bash
mopforge gpu train configs/jobs/tiny_gpu_smoke.json --device cuda --precision bf16
```

The tiny profile is a functional check. Do not use it as performance evidence.

Useful run commands:

```bash
mopforge gpu list
mopforge gpu show <run_id>
mopforge gpu resume <run_id>
mopforge gpu benchmark <run_id>
mopforge gpu compare-runs <run_id> <run_id> --output outputs/comparison.json
```

## Cached Sparse Code-Repair Workflow

Prepare a fixed verified repair split:

```bash
mopforge gpu prepare-efficiency-data \
  --count-per-category 2000 \
  --split-seed 42 \
  --stratify-by bug_type \
  --quality-format fixed_code_xml \
  --verify \
  --overwrite
```

Train a Dense or MoP Full warm source, then build a teacher activation cache:

```bash
mopforge gpu train configs/jobs/100m_mop_full_extended_efficiency.json

mopforge gpu cache-activations configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json \
  --checkpoint <warm_source_checkpoint_or_run_id> \
  --output outputs/code_repair_teacher_topk_cache_manifest.json \
  --teacher-top-k 16 \
  --records-per-shard 512
```

Generate sparse-tail configs:

```bash
mopforge gpu write-warm-sparse-sweep \
  --base-checkpoint <warm_source_checkpoint_or_run_id> \
  --activation-cache-path outputs/code_repair_teacher_topk_cache_manifest.json \
  --dataset-ref <dataset_id@version_id> \
  --dataset-split-id <split_id> \
  --cached-distillation-weight 0.2 \
  --cached-distillation-temperature 2.0 \
  --cached-distillation-top-k 16 \
  --hard-example-replay \
  --hard-example-replay-loss-threshold <teacher_ce_loss_threshold> \
  --hard-example-replay-multiplier 2 \
  --target-eval-loss <predeclared_target_eval_loss> \
  --output-dir configs/jobs/code_repair_cached_sparse
```

Run comparable profiles and gate the claim:

```bash
mopforge gpu compare-runs <dense_run_id> <cached_sparse_run_id> \
  --output outputs/code_repair_comparison.json \
  --output-csv outputs/code_repair_comparison.csv

mopforge gpu gate-efficiency \
  --dense-run <dense_run_id> \
  --sparse-run <cached_sparse_run_id> \
  --output outputs/code_repair_efficiency_gate.json
```

## Current Measured L4 Workflow

The tracked L4 notebook for the current measured report is:

```text
notebooks/colab_l4_verified_code_repair_100m.ipynb
```

It writes a lightweight report compatible with:

```text
reports/verified_code_repair_100m_l4/
```

The report excludes checkpoints, optimizer state, activation caches, token
shards, corpora, and model weights.

## A100 1B Admission

Run a staged probe before any 1B pilot:

```bash
mopforge gpu validate configs/jobs/1b_dense_a100_40gb_probe.json
mopforge gpu estimate configs/jobs/1b_dense_a100_40gb_probe.json
mopforge gpu probe configs/jobs/1b_dense_a100_40gb_probe.json \
  --optimizer-updates 20 \
  --output reports/a100_1b_feasibility_probe/dense_40gb_probe.json
```

Use the matching `80gb` profile on an A100 80 GB. Profiles also exist for
`mop_full` and `cached_adapter_128`. The cached profile requires a real warm
base checkpoint and activation-cache manifest.

Passing requires:

- no OOM,
- finite decreasing loss,
- exact optimizer-update count,
- model-only checkpoint save/load/resume,
- peak reserved VRAM within the hardware-specific gate,
- measured runtime projections.

Colab entrypoint:

```text
notebooks/colab_a100_1b_feasibility_probe.ipynb
```

## H100 2B Readiness

Build tokenizer and packed shards before the H100 probes:

```bash
mopforge tokenizer train-bpe data/code_corpus.jsonl \
  --output-dir data/tokenizer_32k \
  --vocab-size 32768 \
  --text-field text

mopforge gpu pack-corpus data/code_corpus.jsonl \
  --tokenizer-spec data/tokenizer_32k/tokenizer_spec.json \
  --output-dir data/code_tokens_1024 \
  --sequence-length 1024 \
  --split-seed 42 \
  --text-field text
```

Run calibration, 1B admission, and the matching H100 2B probe in order:

```bash
mopforge gpu probe configs/jobs/h100_300m_dense_probe.json \
  --optimizer-updates 20 \
  --output reports/h100_2b_readiness/300m_probe.json

mopforge gpu probe configs/jobs/h100_1b_dense_probe.json \
  --optimizer-updates 20 \
  --output reports/h100_2b_readiness/1b_probe.json
```

The ready-to-run notebook is:

```text
notebooks/colab_h100_2b_readiness.ipynb
```

See [production_2b_readiness.md](production_2b_readiness.md) for model sizes,
distributed pilot commands, post-training, evaluation, export, and claim
requirements.
