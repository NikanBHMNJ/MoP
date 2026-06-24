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
| Split stratification | `bug_type` for code-quality runs |
| Train shuffle seed | `42` |
| Quality format | `raw` or `fixed_code_xml` |
| Target eval loss |  |
| Microsteps |  |
| Optimizer updates |  |
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
mopforge gpu prepare-efficiency-data --count-per-category 100 --split-seed 42 --stratify-by bug_type --quality-format fixed_code_xml

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
| Full held-out eval | every eval lesson measured |  |  |
| Best-checkpoint generation | checkpoint path and step recorded |  |  |
| Ground-truth controls | raw and XML verifier pass are 100% |  |  |
| Category coverage | all five bug categories reported |  |  |
| Target truncation | zero, or explicitly justified |  |  |

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

## Goal 51 1B Admission Addendum

Before a 1B pilot, attach the `mopforge gpu probe` report and fill these measured
fields. Do not copy static estimator values into observed columns.

| Gate | A100 40 GB | A100 80 GB | Observed | Pass |
| --- | ---: | ---: | ---: | --- |
| Optimizer updates | 20-50 | 20-50 |  |  |
| Peak reserved VRAM | <=34 GB | <=68 GB |  |  |
| Finite loss | required | required |  |  |
| Loss decreases | required | required |  |  |
| OOM/allocator retry audit | no OOM | no OOM |  |  |
| Atomic model-only resume | same-batch loss matches | same-batch loss matches |  |  |
| 500-update projection | recorded | recorded |  |  |
| 2,000-update projection | recorded | recorded |  |  |

Record every phase: model/data allocation, forward, backward, optimizer-state
initialization and steady updates, evaluation, checkpoint save/load, and
cleanup. Parameter storage dtype and BF16 autocast compute dtype are separate
facts and must both appear in the report.

For the 100M learning diagnostic, use
`notebooks/colab_l4_goal50_100m_learning_gate.ipynb` before filling this full
comparison. Do not interpret microsteps as optimizer updates, and do not scale
to 1B when the memorization gate fails.

## Goal 52 Scale Addendum

For production-decoder/H100 reports, attach the packed-token manifest, local BPE
spec hash, analytic and runtime parameter counts, scheduler unit, tokens seen,
distributed world size, and checkpoint format. FSDP estimates must state the
declared shard factor; measured per-rank peak allocated and reserved VRAM remain
authoritative.

Do not compare a Dense 2B model with a routed MoP only by total parameters.
Report total parameters, active parameters per token, trainable parameters, and
router/expert balance separately. Before a usability claim, attach standard
code pass@1, contamination evidence, verified repair quality, representative
samples, and the consolidated-checkpoint/Hugging Face export report.

## Interpretation

Use this wording unless the data says otherwise:

```text
This run evaluates whether warm-started sparse MoP, especially cached sparse
teacher-distilled training on a fixed code split, can close the Goal 46
adapter-only loss gap while retaining a measurable efficiency advantage.
It is evidence for the measured axes only and should not be generalized beyond
the tested data, model size, hardware, and seed.
```
