# Production 2B Readiness

This document describes the current production-sized MoP-Forge implementation
path. It does not claim that a 2B model has already trained successfully on
H100 hardware. H100 feasibility is admitted only by measured reports under:

```text
reports/h100_2b_readiness/
```

## Implemented Stack

- `production_decoder_v2` with pre-norm RoPE/RMSNorm blocks, grouped-query
  attention, SwiGLU, PyTorch SDPA, activation checkpointing, and native K/V
  caching.
- Dense, oracle MoP, and learned top-k token-routed MoP feed-forward execution.
- Local byte-level BPE tokenizer training with immutable special-token IDs and
  source hashes.
- Deterministic document split, fixed-length packing, memory-mapped `uint32`
  token shards, and SHA-256 provenance manifests.
- Optimizer-step and token-budget scheduling with exact update/cursor resume.
- Torchrun DDP and FSDP execution with rank-aware samplers and no-sync gradient
  accumulation.
- Distributed Checkpoint model/optimizer shards, atomic sidecars, resume, and
  CPU consolidation to a normal model checkpoint.
- Verified SFT through `GPUTrainer`, cached-reference DPO, and reference-free
  ORPO.
- HumanEval-, MBPP-, and native JSONL evaluation adapters with pass@1, syntax,
  exact match, task artifacts, and contamination auditing.
- Hugging Face Llama export for a Dense model or one named materialized MoP
  expert.

## Tokenizer And Packed Data

Use a trusted, licensed UTF-8 text corpus or JSONL with a `text` field:

```bash
mopforge tokenizer train-bpe data/code_corpus.jsonl \
  --output-dir data/tokenizer_32k \
  --vocab-size 32768 \
  --text-field text

mopforge gpu pack-corpus data/code_corpus.jsonl \
  --tokenizer-spec data/tokenizer_32k/tokenizer_spec.json \
  --output-dir data/code_tokens_1024 \
  --sequence-length 1024 \
  --tokens-per-shard 10000000 \
  --eval-fraction 0.01 \
  --split-seed 42 \
  --text-field text
```

The manifest fixes source hashes, tokenizer identity, split seed, sequence
length, shard hashes, record counts, and packing efficiency. Corpus files,
tokenizer artifacts, and shards remain outside Git.

## Single-H100 Admission

Run the probes in order:

```bash
mopforge gpu probe configs/jobs/h100_300m_dense_probe.json \
  --optimizer-updates 20 \
  --output reports/h100_2b_readiness/300m_probe.json

mopforge gpu probe configs/jobs/h100_1b_dense_probe.json \
  --optimizer-updates 20 \
  --output reports/h100_2b_readiness/1b_probe.json

mopforge gpu probe configs/jobs/h100_2b_dense_80gb_probe.json \
  --optimizer-updates 20 \
  --output reports/h100_2b_readiness/2b_probe.json
```

Select the 94 GB profile only for a detected 86-100 GiB H100. Admission
requires no OOM, finite decreasing loss, model-only checkpoint reload, and peak
reserved VRAM within the profile limit. The probe never silently reduces model
size, context length, microbatch, or accumulation.

Analytic profile sizes:

| Profile | Total parameters | Active parameters |
| --- | ---: | ---: |
| Dense calibration | 304,137,216 | 304,137,216 |
| Dense 1B | 1,015,779,072 | 1,015,779,072 |
| Dense 2B | 2,082,246,912 | 2,082,246,912 |
| Routed MoP | 2,480,265,984 | approximately 1,015,779,072 |

Runtime reports remain authoritative.

## Eight-H100 Pilot

Inspect the launch command:

```bash
mopforge gpu launch-torchrun \
  configs/jobs/h100_2b_dense_fsdp_pilot.json --dry-run
```

Run the printed torchrun command on the eight-GPU node. The pilot profile uses
FSDP, sharded optimizer/model checkpoints, token-unit cosine scheduling, BF16
compute, 1,024-token context, and no quantization.

After training, consolidate on a host with enough RAM and disk:

```bash
mopforge gpu consolidate-checkpoint \
  gpu_runs/<run_id>/checkpoints/checkpoint-step-<step> \
  outputs/2b-consolidated.pt
```

The consolidated file omits optimizer state and is suitable for evaluation,
post-training, or export. Keep it out of Git.

## Preference Post-Training

```bash
mopforge posttrain prepare-preferences \
  data/verified_lessons.jsonl \
  gpu_runs/<run_id>/generation_eval.json \
  data/preferences.jsonl

mopforge posttrain preference configs/posttrain/h100_2b_dpo.json
```

The distributed verified SFT profile is
`configs/jobs/h100_2b_verified_sft_fsdp.json`. Consolidate its best
checkpoint before DPO/ORPO.

## Evaluation And Export

Only run the local verifier on trusted benchmark code:

```bash
mopforge eval contamination data/humaneval.jsonl data/code_corpus.jsonl \
  --format humaneval \
  --output outputs/humaneval_contamination.json

mopforge eval code outputs/2b-consolidated.pt data/humaneval.jsonl \
  outputs/humaneval_eval.json \
  --format humaneval \
  --device cuda

mopforge model export-hf outputs/2b-consolidated.pt outputs/hf-export
```

For routed MoP, pass `--expert repair` or another configured expert to export
one standard dense Llama specialist. Dynamic routing is not represented in that
Hugging Face artifact.

## Claim Boundary

Do not call a model usable because training loss decreases. Publish at least:

- held-out perplexity/loss and exact data/tokenizer identity,
- HumanEval/MBPP or equivalent pass@1 with contamination evidence,
- verified repair pass and syntax rates on a separate repair set,
- checkpoint-resume equivalence and update/token counts,
- peak allocated/reserved VRAM and tokens/sec on named hardware,
- a second seed for strong comparative claims,
- representative generated samples and failure categories.

No tracked H100 report currently contains measured values.
