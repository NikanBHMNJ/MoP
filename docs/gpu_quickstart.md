# GPU Quickstart

MoP-Forge now includes a serious single-GPU research beta for tiny-to-small MoP
experiments and validated large-job profiles. It is not yet a fully production
distributed LLM training framework.

CPU-only development path:

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
```

CUDA path, when PyTorch CUDA is installed:

```bash
mopforge runtime detect
mopforge gpu train configs/jobs/tiny_gpu_smoke.json --device cuda --precision bf16
```

The tiny smoke profile falls back to CPU when CUDA is unavailable. Real GPU
performance is only validated on the user's hardware.

Useful follow-up commands:

```bash
mopforge gpu list
mopforge gpu show <run_id>
mopforge gpu resume <run_id>
mopforge gpu benchmark <run_id>
```

Cached sparse distillation path for a code dataset:

```bash
mopforge gpu prepare-efficiency-data --count-per-category 100 --split-seed 42
mopforge gpu train configs/jobs/100m_mop_full_extended_efficiency.json

mopforge gpu cache-activations configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json \
  --checkpoint <mop_full_run_id_or_checkpoint> \
  --output outputs/code_warm_sparse_teacher_topk_cache_manifest.json \
  --teacher-top-k 16 \
  --records-per-shard 512

mopforge gpu write-warm-sparse-sweep \
  --base-checkpoint <mop_full_run_id_or_checkpoint> \
  --activation-cache-path outputs/code_warm_sparse_teacher_topk_cache_manifest.json \
  --dataset-ref <dataset_id@version_id> \
  --dataset-split-id <split_id> \
  --cached-distillation-weight 0.2 \
  --cached-distillation-temperature 2.0 \
  --cached-distillation-top-k 16 \
  --hard-example-replay \
  --hard-example-replay-loss-threshold <teacher_ce_loss_threshold> \
  --hard-example-replay-multiplier 2 \
  --target-eval-loss <dense_or_mop_full_target_loss>
```

The cached sparse configs train the tail from cached hidden states, can offload
unused frozen backbone modules from CUDA, save the best eval-loss checkpoint,
optionally replay high-loss cached examples, and report distillation/offload,
hard replay, plus time/tokens-to-target-loss metadata in `metrics.json` and
`compare-runs` outputs.

For the next small-model code-quality run, keep the same fixed split discipline
but frame verified repair targets as narrow XML blocks:

```bash
mopforge gpu prepare-efficiency-data \
  --count-per-category 10 \
  --split-seed 42 \
  --stratify-by bug_type \
  --quality-format fixed_code_xml
```

That mode writes lessons whose supervised target is
`<fixed_code>...</fixed_code>`. It is intended for code repair, short
completion, and test-error-conditioned fixing experiments where syntax pass,
verifier pass, and exact-match quality are measured alongside VRAM, throughput,
loss, and checkpoint size.

For the full L4 quality comparison, run
`notebooks/colab_l4_goal49_verified_code_quality_report.ipynb`. It compares
Warm Adapter/Norm/Head 128 with cached Adapter 128 and tail-only LoRA rank 8/16.
Tail-only LoRA places the trainable deltas after the activation-cache boundary,
allowing the frozen backbone to remain off CUDA during sparse training.

Cached runs now restore the full model only after training to generate and
verify code samples. Those samples are written to `generation_eval.json`; the
restoration occurs after cached-tail VRAM metrics are captured.

Before a 1B quality run, execute
`notebooks/colab_l4_goal50_100m_learning_gate.ipynb`. Its config uses full
held-out loss evaluation, deterministic epoch reshuffling, 1,000 optimizer
updates, generation from the best eval-loss checkpoint, all five bug
categories, and raw/XML ground-truth controls. The generated report records
microsteps and optimizer updates separately and blocks scaling when the
memorization thresholds fail.

The first measured Goal 50 gate exposed an EOS prompt-boundary mismatch. After
that fix, the rerun reached `100%` train and held-out XML completion, syntax,
verifier, and exact match. The full 100M comparison is now permitted; 1B remains
gated on its quality and efficiency evidence.

After a passing gate, use
`notebooks/colab_l4_goal50_100m_quality_comparison.ipynb` for the full 100M
comparison. Its shared target is predeclared as `TARGET_EVAL_LOSS=0.85`, based
on the Goal 49 Dense best loss of `0.8022`. The notebook uses
2,000 optimizer updates per enabled profile, 10,000 balanced verified lessons,
full held-out loss, a five-category stratified generation subset, and an
`acceptance_gates.json` claim boundary.

The first measured full comparison passes after correcting a report-only
metadata fallback: Cached Adapter/Norm/Head 128 reached `88.0%` verifier/exact
match, `8.35x` Dense throughput, and `31.70x` lower peak reserved VRAM. Cached
Tail-Only LoRA Rank 8 reached the same quality with `6.70x` throughput and
`22.83x` lower peak reserved VRAM.
