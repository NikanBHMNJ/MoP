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
