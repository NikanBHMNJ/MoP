# Warm Sparse GPU Efficiency Comparison Template

Use this template for the next 100M Dense vs full-MoP vs warm-sparse
comparison. Fill it only with measured run outputs. Leave cells blank until the
corresponding run exists.

## Purpose

Test whether warm-started sparse MoP can reduce loss versus the Goal 46
adapter-only result while preserving a measurable efficiency advantage over
Dense.

Do not use this report to claim MoP superiority unless the quality and
efficiency gates both pass.

## Runs

| Role | Config | Run ID | Checkpoint |
| --- | --- | --- | --- |
| Dense baseline extended | `configs/jobs/100m_dense_extended_efficiency.json` |  |  |
| MoP full warm-start source | `configs/jobs/100m_mop_full_extended_efficiency.json` |  |  |
| Warm adapters 64 | `configs/jobs/100m_mop_warm_adapters_64_colab_efficiency.json` |  |  |
| Warm adapters norm/head 64 | `configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json` |  |  |
| Core frozen quality | `configs/jobs/100m_mop_core_frozen_quality_colab_efficiency.json` |  |  |
| Cached warm adapter tail + teacher top-k KL | generated warm sparse cache config |  |  |
| Cached warm adapter norm/head 128 + teacher top-k KL | Goal 49 notebook config |  |  |
| Cached tail-only LoRA rank 8/16 + teacher top-k KL | Goal 49 notebook config |  |  |
| Warm routed LoRA rank 4/8/16 | generated warm sparse sweep config |  |  |
| Routed FFN top-1 | `configs/jobs/100m_mop_routed_ffn_expert_efficiency.json` |  |  |

## Dataset And Budget

| Field | Value |
| --- | --- |
| Dataset ref |  |
| Split ID |  |
| Split seed | `42` |
| Quality format | `raw` or `fixed_code_xml` |
| Target eval loss |  |
| Max steps | `2000` |
| Gradient accumulation | `8` |
| Micro batch size | `1` |
| Max sequence length | `1024` |
| Precision |  |
| Device |  |

## Results

| Model | Train loss | Eval loss | Best eval loss | Time-to-target sec | Tokens-to-target | Tokens/sec | Peak allocated VRAM | Peak reserved VRAM | Target peak reserved VRAM | Trainable ratio | Active trainable ratio | Checkpoint size | Distill top-k | Offloaded params | Hard replayed | Exact match | Verifier pass | Syntax pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MoP Full |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Warm Adapter 64 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Warm Adapter Norm/Head 64 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Core Frozen |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cached Warm Adapter Norm/Head 64 + KL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cached Warm Adapter Norm/Head 128 + KL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cached Tail-Only LoRA Rank 8/16 + KL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Warm Routed LoRA |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Routed FFN Top-1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Commands

```powershell
mopforge gpu prepare-efficiency-data --count-per-category 100 --split-seed 42 --quality-format fixed_code_xml

mopforge gpu train configs/jobs/100m_dense_extended_efficiency.json
mopforge gpu train configs/jobs/100m_mop_full_extended_efficiency.json

mopforge gpu cache-activations `
  configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json `
  --checkpoint <mop_full_run_id_or_checkpoint> `
  --output outputs/warm_sparse_teacher_topk_cache_manifest.json `
  --teacher-top-k 16 `
  --records-per-shard 512

mopforge gpu write-warm-sparse-sweep `
  --base-checkpoint <mop_full_run_id_or_checkpoint> `
  --dataset-ref <dataset_id@version_id> `
  --dataset-split-id <split_id> `
  --activation-cache-path outputs/warm_sparse_teacher_topk_cache_manifest.json `
  --cached-distillation-weight 0.2 `
  --cached-distillation-temperature 2.0 `
  --cached-distillation-top-k 16 `
  --hard-example-replay `
  --hard-example-replay-loss-threshold <teacher_ce_loss_threshold> `
  --hard-example-replay-multiplier 2 `
  --target-eval-loss <dense_or_mop_full_target_loss> `
  --output-dir configs/jobs/warm_sparse_sweep

mopforge gpu compare-runs <dense_run_id> <mop_full_run_id> <sparse_run_id> `
  --output outputs/warm_sparse_efficiency_comparison.json `
  --output-csv outputs/warm_sparse_efficiency_comparison.csv

mopforge gpu gate-efficiency `
  --dense-run <dense_run_id> `
  --sparse-run <sparse_run_id> `
  --output outputs/warm_sparse_gate_report.json
```

## Claim Gate

Record the gate result here:

| Gate | Required result | Observed result | Pass |
| --- | --- | --- | --- |
| Eval loss | within configured delta of Dense |  |  |
| Generated-code verifier pass | within configured delta of Dense |  |  |
| Syntax/compile pass | not materially worse than Dense |  |  |
| Time-to-target-loss | improves or matches Dense/MoP Full |  |  |
| Tokens/sec | not materially worse than Dense |  |  |
| Peak allocated VRAM | improves named target axis |  |  |
| Peak reserved VRAM | improves named target axis |  |  |
| Frozen-backbone offload | cached run records offloaded params |  |  |
| Teacher KL | cached run records teacher top-k metadata |  |  |
| Hard-example replay | cached run records replay threshold and count |  |  |
| Trainable ratio | improves named target axis |  |  |
| Checkpoint size | improves named target axis |  |  |

Same-quality sparse efficiency requires both quality and efficiency evidence.
Any `3x` to `50x` claim must name the exact axis: peak VRAM, trainable
parameters, checkpoint delta, cached-tail training time, or active expert
compute.

Use `fixed_code_xml` for small-model code-repair quality runs. It frames
verified targets as `<fixed_code>...</fixed_code>` so generated samples can be
evaluated as narrow repair/completion outputs rather than free-form code.

Use `lora_tail_only=true` for cached LoRA comparisons. This places LoRA after
the cached boundary; ordinary routed LoRA remains inside transformer blocks and
is not a valid frozen-backbone cached-tail profile. Cached-run generation is a
post-training quality evaluation and must not be counted as cached-tail VRAM.

## Interpretation

Use this wording unless the data says otherwise:

```text
This run evaluates whether warm-started sparse MoP, especially cached sparse
teacher-distilled training on a fixed code split, can close the Goal 46
adapter-only loss gap while retaining a measurable efficiency advantage.
It is evidence for the measured axes only and should not be generalized beyond
the tested data, model size, hardware, and seed.
```
