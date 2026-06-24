# Goal 52 Production And 2B Readiness

Goal 52 adds an end-to-end implementation path for production-sized Dense and
MoP experiments. It does not claim that the 2B profiles have already trained
successfully on H100 hardware. The tracked notebook and report schema enforce
that distinction.

## Implemented Stack

- `production_decoder_v2`: pre-norm RoPE/RMSNorm decoder blocks, grouped-query
  attention, SwiGLU, SDPA, activation checkpointing, and native K/V caching.
- Dense, oracle MoP, and learned top-k token-routed MoP feed-forward execution.
- Local byte-level BPE tokenizer training with a serializable tokenizer spec.
- Deterministic document split, fixed-length packing, memory-mapped `uint32`
  shards, and SHA-256 provenance manifests.
- Optimizer-step or token-budget scheduling with exact update/cursor resume.
- Torchrun DDP and FSDP execution with rank-aware samplers and no-sync gradient
  accumulation.
- Distributed Checkpoint model/optimizer shards, atomic sidecars, resume, and
  CPU consolidation to a normal model checkpoint.
- Verified supervised code training through `GPUTrainer`, cached-reference DPO,
  and reference-free ORPO.
- HumanEval-, MBPP-, and native JSONL evaluation adapters with pass@1, syntax,
  exact match, task artifacts, and contamination auditing.
- Hugging Face Llama export for a Dense model or one named materialized MoP
  expert.

## Build Tokenizer And Packed Data

Use a trusted, licensed UTF-8 text corpus or JSONL with a `text` field:

```bash
mopforge tokenizer train-bpe data/code_corpus.jsonl \
  --output-dir data/goal52_tokenizer \
  --vocab-size 32768 \
  --text-field text

mopforge gpu pack-corpus data/code_corpus.jsonl \
  --tokenizer-spec data/goal52_tokenizer/tokenizer_spec.json \
  --output-dir data/goal52_code_tokens \
  --sequence-length 1024 \
  --tokens-per-shard 10000000 \
  --eval-fraction 0.01 \
  --split-seed 42 \
  --text-field text
```

The manifest fixes source hashes, tokenizer identity, split seed, sequence
length, shard hashes, record counts, and packing efficiency. Corpus files,
tokenizer artifacts, and shards remain outside Git.

## H100 Admission

Run the profiles in order. The notebook automates the same gates:

```bash
mopforge gpu probe configs/jobs/goal52_300m_dense_h100_probe.json \
  --optimizer-updates 20 \
  --output reports/goal52_h100_2b_readiness/300m_probe.json

mopforge gpu probe configs/jobs/goal52_1b_dense_h100_probe.json \
  --optimizer-updates 20 \
  --output reports/goal52_h100_2b_readiness/1b_probe.json

# Select only the tier matching torch.cuda.get_device_properties(0).total_memory.
mopforge gpu probe configs/jobs/goal52_2b_dense_h100_80gb_probe.json \
  --optimizer-updates 20 \
  --output reports/goal52_h100_2b_readiness/2b_probe.json
```

The 94 GB profile is selected only for a detected 86-100 GiB H100. Admission
requires no OOM, finite decreasing loss, a passing model-only checkpoint reload,
and peak reserved VRAM within the profile limit. The probe never silently
reduces context, model size, microbatch, or accumulation.

The exact analytic parameter counts are:

| Profile | Total parameters | Active parameters |
| --- | ---: | ---: |
| Dense calibration | 304,137,216 | 304,137,216 |
| Dense 1B | 1,015,779,072 | 1,015,779,072 |
| Dense 2B | 2,082,246,912 | 2,082,246,912 |
| Routed MoP | 2,480,265,984 | approximately 1,015,779,072 |

The MoP active value is an analytic planning estimate for one selected FFN
expert per token. Runtime reports remain authoritative.

## Eight-H100 Pilot And Resume

Inspect the launch command:

```bash
mopforge gpu launch-torchrun \
  configs/jobs/goal52_2b_dense_8xh100_fsdp_pilot.json --dry-run
```

Run the printed torchrun command on the eight-GPU node. The 500-update profile
uses FSDP, sharded optimizer/model checkpoints, token-unit cosine scheduling,
BF16 compute, FP32 initial parameter storage, 1,024-token context, and no
quantization. Resume points to the sharded checkpoint directory.

After training, consolidate on a host with enough RAM and disk:

```bash
mopforge gpu consolidate-checkpoint \
  gpu_runs/<run_id>/checkpoints/checkpoint-step-<step> \
  outputs/goal52-2b-consolidated.pt
```

The consolidated file omits optimizer state and is suitable for evaluation,
post-training, or export. Keep it out of Git.

## Preference Post-Training

Turn failed generation records into verified target preferences:

```bash
mopforge posttrain prepare-preferences \
  data/verified_lessons.jsonl \
  gpu_runs/<run_id>/generation_eval.json \
  data/goal52_preferences.jsonl

mopforge posttrain preference configs/posttrain/goal52_dpo.json
```

The distributed verified SFT profile is
`configs/jobs/goal52_2b_verified_sft_8xh100_fsdp.json`; it warm-starts from the
consolidated pretraining checkpoint and writes sharded best/resume checkpoints.
Consolidate its best checkpoint before DPO/ORPO.

The preference config names a consolidated source checkpoint, the preference
JSONL, `objective` (`dpo` or `orpo`), update budget, precision, and output
directory. DPO caches reference chosen/rejected log probabilities once and
removes the reference model before policy optimization. ORPO requires no
reference model. This preference trainer is single-accelerator; distributed SFT
uses the main `GPUTrainer`.

## Evaluation And Export

Only run the local verifier on trusted benchmark code:

```bash
mopforge eval contamination data/humaneval.jsonl data/code_corpus.jsonl \
  --format humaneval \
  --output outputs/humaneval_contamination.json

mopforge eval code outputs/goal52-2b-consolidated.pt data/humaneval.jsonl \
  outputs/humaneval_eval.json \
  --format humaneval \
  --device cuda

mopforge model export-hf outputs/goal52-2b-consolidated.pt outputs/goal52-hf
```

For routed MoP, pass `--expert repair` (or another configured expert) to export
one standard dense Llama specialist. Dynamic routing is not represented in that
Hugging Face artifact.

## Claim Boundary

Do not call the model usable because training loss decreases. Publish at least:

- held-out perplexity/loss and exact data/tokenizer identity,
- HumanEval/MBPP or equivalent pass@1 with contamination evidence,
- verified task pass and syntax rates on a separate repair set,
- checkpoint-resume equivalence and update/token counts,
- peak allocated/reserved VRAM and tokens/sec on named hardware,
- a second seed for strong comparative claims,
- representative generated samples and failure categories.

No tracked Goal 52 H100 report currently contains measured values.
