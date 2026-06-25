# MoP-Forge

**Version:** `0.46.0`
**Status:** production-oriented research framework with measured verified-code
repair evidence and scale-readiness tooling

MoP-Forge is a local-first training and evaluation framework for
Mixture-of-Parameters (MoP), dense baselines, cached sparse training, routed
experts, and verified code-model workflows. The repository is organized around
reproducible evidence: a public claim must point to a report folder, fixed
data split, hardware target, metrics, limitations, and a passing claim card.

The current supported release surface is `0.46.0`. Earlier report snapshots and
historical milestone narratives have been removed from the primary repo surface.
Current evidence lives under [reports/](reports/).

MoP-Forge is not a managed cloud training service and does not claim a
Qwen-class, frontier-class, or generally usable 2B model. It provides the
framework, profiles, admission probes, export path, and claim gates needed to
measure those claims honestly.

## Current Evidence

The current measured report is:

- [Verified code repair 100M L4](reports/verified_code_repair_100m_l4/README.md)

On a fixed 10,000-lesson verified code-repair split, the cached sparse profiles
preserved task quality while improving the measured efficiency axes:

| Profile | Verifier | Exact | Tokens/sec | Peak reserved VRAM | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 82.4% | 82.4% | 8,348.76 | 1.9180 GB | 987.23 MB |
| Cached Adapter/Norm/Head 128 | 88.0% | 88.0% | 69,671.29 | 0.0605 GB | 7.74 MB |
| Cached Tail-Only LoRA Rank 8 | 88.0% | 88.0% | 55,928.46 | 0.0840 GB | 8.12 MB |

This supports a narrow A2/P2 claim: measured verified code-repair efficiency on
the named dataset split, seed, and L4 hardware. It does not support broad code
generation, product-beta, frontier-model, or paper-ready claims.

Validate the claim:

```bash
mopforge claim scaffold \
  --report-dir reports/verified_code_repair_100m_l4 \
  --claim-statement "MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4 report split." \
  --academic-level A2 \
  --product-level P2 \
  --output reports/verified_code_repair_100m_l4/claim_card.json \
  --validation-output reports/verified_code_repair_100m_l4/claim_validation.json

mopforge claim validate reports/verified_code_repair_100m_l4/claim_card.json
```

Scale-readiness report targets:

- [A100 1B feasibility probe](reports/a100_1b_feasibility_probe/README.md)
- [H100 2B readiness](reports/h100_2b_readiness/README.md)

These directories currently contain admission schemas and instructions. They
are not measured A100/H100 feasibility claims until hardware reports are added.

## Implemented Framework

MoP-Forge includes:

- production decoder profiles with RoPE, RMSNorm, grouped-query attention,
  SwiGLU, PyTorch SDPA, activation checkpointing, and incremental K/V cache,
- Dense, full-MoP, routed FFN, warm sparse, cached sparse, adapter, norm/head,
  and cache-compatible tail-only LoRA training profiles,
- local byte-level BPE tokenizer training and deterministic memory-mapped token
  shards,
- single-device CUDA/BF16 training plus torchrun DDP/FSDP execution,
- exact optimizer-step, token-budget, eval, save, and resume accounting,
- Distributed Checkpoint save/resume and model-only consolidation,
- activation-cache training that keeps unused frozen backbone modules off CUDA
  during cached sparse-tail training,
- teacher top-k distillation, hard-example replay, and verified fixed-code XML
  target framing,
- generated-code evaluation with exact match, syntax pass, verifier pass,
  ground-truth controls, per-category failures, and generated samples,
- A100/H100 admission probes with allocator, host memory, checkpoint, OOM, and
  runtime projection telemetry,
- DPO and ORPO preference post-training,
- HumanEval-, MBPP-, and native JSONL code evaluation with contamination audit,
- Hugging Face Llama-compatible export for Dense checkpoints or one materialized
  MoP expert,
- executable academic/product claim gates through `mopforge claim`.

## Install

```bash
pip install -e .[dev]
mopforge doctor
mopforge runtime detect
```

Optional extras:

```bash
pip install -e .[dev,gpu,hf]
```

## Core Commands

Validate a GPU profile:

```bash
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/tiny_gpu_smoke.json
```

Run a local training profile:

```bash
mopforge gpu train configs/jobs/tiny_gpu_smoke.json --device cuda --precision bf16
```

Compare completed GPU runs:

```bash
mopforge gpu compare-runs <dense_run_id> <sparse_run_id> \
  --output outputs/gpu_efficiency_comparison.json \
  --output-csv outputs/gpu_efficiency_comparison.csv
```

Gate a sparse-efficiency claim:

```bash
mopforge gpu gate-efficiency \
  --dense-run <dense_run_id> \
  --sparse-run <sparse_run_id> \
  --output outputs/gpu_efficiency_gate_report.json
```

Run an A100 admission probe:

```bash
mopforge gpu probe configs/jobs/1b_dense_a100_40gb_probe.json \
  --optimizer-updates 20 \
  --output reports/a100_1b_feasibility_probe/dense_40gb_probe.json
```

Build tokenizer and packed shards for production-sized runs:

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

## Notebooks

Tracked notebooks are current operational entrypoints:

- [L4 verified code repair 100M](notebooks/colab_l4_verified_code_repair_100m.ipynb)
- [A100 1B feasibility probe](notebooks/colab_a100_1b_feasibility_probe.ipynb)
- [H100 2B readiness](notebooks/colab_h100_2b_readiness.ipynb)

Notebook outputs must remain lightweight. Do not commit checkpoints, optimizer
state, token shards, activation caches, corpora, or model weights.

## Documentation

- [Documentation index](docs/README.md)
- [GPU quickstart](docs/gpu_quickstart.md)
- [Production 2B readiness](docs/production_2b_readiness.md)
- [Efficiency report template](docs/efficiency_report_template.md)
- [Academic claim standard](docs/academic_claim_standard.md)
- [Startup and product claim standard](docs/startup_product_claim_standard.md)
- [Claim readiness template](reports/claim_readiness_template/)
- [Reports index](reports/README.md)

## Validation

Current release checks:

```bash
python -m pytest -q
python scripts/release_check.py --quick-examples
git diff --check
```

The release check also verifies that the claim CLI is present.

## Claim Boundary

Use exact, scoped wording. Current allowed public wording is:

```text
MoP-Forge is a production-oriented research framework for measuring dense,
sparse, cached, and routed code-model training workflows. Its current measured
result is narrow verified 100M code-repair efficiency on the fixed L4 report
split.
```

Blocked until measured:

- Qwen-class or frontier-class model quality,
- broad code-generation quality,
- production managed-service readiness,
- guaranteed 1B/2B training on arbitrary hardware,
- same-quality sparse superiority beyond the named report,
- customer-proven cost savings.
