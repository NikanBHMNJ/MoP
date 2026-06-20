# Implementation Plan: Lower Loss And GPU Use Without Quantization

## Implementation Status

Implemented in this branch:

- Warm-start sparse GPU training via `resume_model_only`.
- Sparse trainable policy support via `train_norm` and
  `adapters_norm_head`.
- Configurable MoP module names and `always_include_core=false` support for
  new efficiency configs without a redundant `module:core` block.
- Frozen-prefix execution metadata and no-grad execution for frozen
  embeddings/shared core, plus frozen module-bank no-grad when the routed
  module bank is frozen and no trainable upstream path needs input gradients.
- Real non-reentrant PyTorch activation checkpointing across dense, shared,
  and routed transformer blocks, with applied-block metadata.
- Cached frozen-prefix activation training:
  - `mopforge gpu cache-activations`
  - `activation_cache_path` GPU training mode
  - cached hidden-state DataLoaders
  - original attention masks, labels, target modules, source IDs, checkpoint
    hashes, and config hashes in the cache metadata.
  - hard rejection when any encoded prefix, module bank, or routed expert block
    remains trainable.
- Trainable-only sparse checkpoints with base-checkpoint references and
  compatible model-only restore.
- Extended efficiency metrics:
  - active trainable parameter estimate
  - active trainable parameter ratio
  - shared frozen params
  - routed module params
  - active expert count and compute ratios
  - estimated active/backward FLOP ratios
  - frozen-prefix metadata
- Cached-tail throughput metrics:
  - original-token-equivalent tokens/sec
  - cached-hidden-state steps/sec
- Routed-FFN execution with a configurable shared trunk, separately named
  expert blocks, physical top-k token/example dispatch, router gradients, and
  per-block routing metadata.
- Dense-to-routed warm-start translation that clones each learned dense FFN
  into every corresponding expert and preserves initial logits to numerical
  precision.
- Zero-initialized, module-routed low-rank deltas inside attention Q/K/V,
  attention output, and FFN up/down projections, plus rank `4/8/16` sweeps.
- Optional GPU trainer generated-code evaluation metrics.
- Separate generated-code exact-match and verifier-pass rates for quality
  gating.
- Machine-readable efficiency acceptance gates via
  `mopforge gpu gate-efficiency`.
- Warm sparse sweep generation via
  `mopforge gpu write-warm-sparse-sweep`, covering bottleneck sweeps and
  learning-rate sweeps while preserving the same token budget and fixed eval
  seed.
- Serious coding dataset preparation via
  `mopforge gpu prepare-efficiency-data`, including dataset fingerprints,
  versioned manifests, deterministic train/eval/test splits, and materialized
  split JSONL files.
- Exact registered split reuse via `dataset_split_id`, preventing GPU runs
  from re-splitting the fixed held-out eval bucket.
- Optional early-stopping patience/min-delta tracking, disabled by default for
  fair fixed-token-budget comparisons.
- Warm sparse, core-frozen, routed-FFN, and low-rank-delta GPU config
  templates under `configs/jobs/`.
- Extended 2,000-step dense and full-MoP baselines with matching schedules,
  fixed seeds, generated-code evaluation, and architecture-compatible warm
  starts.
- A warm sparse comparison template at
  `docs/warm_sparse_efficiency_comparison_template.md`.
- Focused regression tests in
  `tests/test_gpu_sparse_efficiency_features.py`.

Still requires real GPU experiment evidence before claiming success:

- Run the new warm-start profiles against a real learned 100M MoP-full
  checkpoint.
- Compare dense, MoP full, warm adapter, warm adapter norm/head, core-frozen,
  and cached-tail runs with the same token budget.
- Accept or reject quality based on eval loss and generated-code pass rate.

## Objective

Reduce the high Goal 46 training/eval loss while also lowering GPU memory use.
The target is a 1.5x to 2x loss reduction from the current sparse MoP result,
without quantization, without reducing output quality, and without giving up
throughput.

Practical target from the report:

- Current MoP Adapter-Only eval loss: `5.1653`
- 1.5x loss target: `<= 3.44`
- 2x loss target: `<= 2.58`
- Dense eval reference: `3.1705`
- MoP Full eval reference: `3.1691`
- Dense peak reserved VRAM: `1.9844 GB`
- MoP Adapter-Only peak reserved VRAM: `0.4961 GB`

The first realistic milestone is to bring sparse MoP close to dense loss while
staying materially below dense VRAM. The stretch milestone is to beat dense loss
with lower VRAM.

## 3x To 50x Efficiency Reality Check

The phrase "3x to 50x lower GPU usage" needs to be split into measurable axes.
Some are realistic immediately; some require architecture changes.

| Efficiency axis | Goal 46 result | Realistic next target | Path |
| --- | ---: | ---: | --- |
| Peak reserved VRAM | adapter-only is already about 4x below dense | 3x to 5x lower than dense | warm-start adapters, frozen-prefix no-grad |
| Trainable parameters | adapter-only is about 1187x below dense | 100x to 1000x lower | adapters/norm/head or LoRA-style deltas |
| Optimizer state memory | scales with trainable params | 100x+ lower | trainable-only optimizer already helps |
| Checkpoint delta size | current adapter checkpoint still stores full model | 50x to 1000x lower | trainable-only sparse checkpoints |
| Active parameter compute | currently still effectively 1.0 active ratio | 3x to 10x lower first | expert-heavy MoP and top-k routed blocks |
| End-to-end training GPU time | not yet improved for same quality | 3x to 50x only for adapter/head sweeps | cached frozen-prefix activations |

Important conclusion:

The current 100M MoP architecture is still shared-core-heavy. Its shared
transformer core has about `85M` parameters, while the module bank is only about
`9.45M`. That means even perfect routing cannot deliver 10x to 50x active
compute savings yet, because almost all useful compute remains in the shared
core. To reach 10x to 50x active GPU efficiency, most parameters must move into
routed expert paths or the frozen shared prefix must be cached.

Therefore the plan has two tracks:

1. Near-term quality-preserving efficiency: warm-started sparse training,
   frozen-prefix no-grad, and trainable-only checkpoints. This should target
   3x to 5x lower peak VRAM and 50x+ lower checkpoint deltas.
2. True MoP GPU-efficiency research: expert-heavy routed transformer blocks and
   activation caching. This is the path toward 10x to 50x lower active training
   cost without quantization.

## Inspection Summary

Report inspected:

- `reports/goal46_gpu_efficiency/README.md`
- `reports/goal46_gpu_efficiency/100m_efficiency_comparison.csv`
- `reports/goal46_gpu_efficiency/runs/*/metrics.json`
- `reports/goal46_gpu_efficiency/runs/*/config.json`

Relevant implementation inspected:

- `mopforge/gpu/trainer.py`
- `mopforge/gpu/config.py`
- `mopforge/gpu/data.py`
- `mopforge/gpu/mop_execution.py`
- `mopforge/training/parameter_policy.py`
- `mopforge/models/tiny_mop.py`
- `mopforge/models/fast_adapters.py`
- `mopforge/models/architectures.py`

Findings:

1. The current adapter-only run trains from scratch while freezing embeddings,
   shared transformer core, module blocks, final norm, and LM head.
2. Adapter-only trained only `80,688` parameters out of `95,773,488`
   parameters, a trainable ratio of `0.000842`.
3. The run used only `200` global steps with gradient accumulation `8`, so it
   had only `25` optimizer steps.
4. The dataset in the report had only `50` lesson records, split into `40`
   train and `10` eval examples.
5. At inspection time, `activation_checkpointing` was metadata only. The
   implementation now executes model-native non-reentrant checkpointing.
6. The MoP architecture includes a `module:core` block inside `module_bank`
   even though it already has `shared_core`. This adds parameters and makes the
   active module accounting less clean.
7. The active parameter estimate stayed at `1.0` for the Goal 46 runs. That
   means the experiment proved sparse trainable parameters, but not yet sparse
   active execution.
8. The checkpoint size is still large for adapter-only because checkpoints save
   the whole model state, including frozen parameters.

Key conclusion:

Adapter-only from a random frozen base is not a fair quality test. Adapters are
for adapting a useful base. To keep quality while reducing GPU use, MoP-Forge
needs warm-started sparse training plus freeze-aware execution.

## Strategy

Use a three-level efficiency ladder:

1. Warm-started adapters plus norm/head.
2. Core-frozen MoP with trainable modules and adapters.
3. Full MoP only as the quality ceiling and warm-start source.

This avoids quantization and keeps BF16/TF32 execution. The improvement comes
from better initialization, more useful trainable capacity, less autograd work
through frozen blocks, and leaner sparse checkpoints.

## Phase 1: Warm-Started Sparse MoP

Problem:

The current adapter-only job freezes a random base. This explains the high
loss. The adapter path should start from a dense or MoP-full checkpoint whose
embeddings, shared core, module blocks, and LM head already learned the data
distribution.

Implementation:

1. Add `resume_model_only: bool = False` to `GPUTrainingConfig`.
2. In `GPUTrainer.load_checkpoint`, support restoring model weights without
   restoring optimizer, scheduler, scaler, or RNG state when `resume_model_only`
   is true.
3. Allow loading a full MoP checkpoint and then applying a different trainable
   policy for the new sparse run.
4. Add config templates:
   - `configs/jobs/100m_mop_warm_adapters_efficiency.json`
   - `configs/jobs/100m_mop_warm_adapters_norm_head_efficiency.json`
   - `configs/jobs/100m_mop_core_frozen_quality_efficiency.json`

Recommended experiment:

```bash
mopforge gpu train configs/jobs/100m_mop_full_colab_efficiency.json

mopforge gpu train configs/jobs/100m_mop_warm_adapters_norm_head_efficiency.json

mopforge gpu compare-runs <dense_run_id> <mop_full_run_id> <warm_adapter_run_id> \
  --output outputs/warm_sparse_efficiency_comparison.json
```

Expected result:

- Eval loss should move toward dense/full MoP because the frozen base is no
  longer random.
- VRAM should remain far below full training because most parameters are frozen.
- Throughput should stay above dense because the optimizer and backward pass are
  sparse.

## Phase 2: Better Sparse Policy, Still Tiny Trainable Ratio

Problem:

`adapters_only` with bottleneck `16` gives only `80,688` trainable parameters.
That is extremely efficient but too small for same-quality training from this
setup.

Implementation:

1. Add `train_norm: bool = False` to `TrainableParameterPolicy`.
2. Add a policy mode or metadata-compatible behavior for:
   - `adapters_norm_head`
   - or `adapters_only` with `metadata.train_norm=true` and
     `metadata.train_lm_head=true`
3. Increase adapter bottleneck from `16` to a sweep of `64`, `128`, and `256`.
4. Keep LM head training optional. Train it only in the first sparse run after
   warm-start, then compare against frozen-head adapters.

Why this helps:

- LayerNorm and LM head add small parameter counts compared with the full
  100M-class model.
- Adapter bottleneck `64` or `128` is still tiny relative to dense training.
- This gives the sparse path enough capacity to lower loss without turning into
  full fine-tuning.

Initial target:

- Trainable ratio: below `0.01`
- Peak reserved VRAM: below `1.0 GB`
- Tokens/sec: at least dense speed, ideally `> 1.5x` dense
- Eval loss: `<= 3.44` first, then close to dense `3.17`

## Phase 3: Freeze-Aware Execution To Reduce VRAM

Problem:

Freezing parameters reduces optimizer state and gradients, but the model should
also make sure frozen prefixes do not keep unnecessary autograd state.

Implementation:

1. In `TinyMoPCausalTransformer.forward`, detect when embeddings and
   `shared_blocks` are fully frozen.
2. If they are frozen, run the embedding/shared-core prefix under
   `torch.no_grad()` and detach the hidden states before trainable modules or
   adapters.
3. Keep normal autograd behavior when any prefix parameter is trainable.
4. Record metadata:
   - `frozen_prefix_no_grad_enabled`
   - `frozen_prefix_param_count`
   - `frozen_prefix_activation_detached`
5. Add tests showing adapter/core-frozen policies still backprop through
   trainable modules/adapters but not through frozen core params.

Why this helps:

- It reduces activation memory for core-frozen and adapter-only runs.
- It should not reduce quality because frozen parameters would not update
  anyway.
- It should not reduce speed; it may improve speed by reducing autograd work.

## Phase 3B: Cached Frozen-Prefix Training For 10x To 50x Adapter Sweeps

Problem:

Even if the shared core is frozen, normal training still runs the shared core
forward pass every step. That preserves quality, but it leaves a lot of GPU
compute on the table.

For adapter-only, norm/head, and other post-core sparse updates, the frozen base
hidden states can be computed once and reused. Since the base is frozen, this is
mathematically equivalent when dropout is disabled or the model is in a stable
cache-generation mode.

Implementation:

1. Add a `mopforge gpu cache-activations` helper.
2. Load the warm base checkpoint.
3. Run the frozen prefix once over the train/eval data:
   - token embeddings
   - positional embeddings
   - shared transformer core
   - routed frozen module blocks if they are frozen
4. Store compact BF16 activation records with:
   - hidden states
   - attention mask
   - labels
   - target modules
   - source example ID
   - base checkpoint hash
   - architecture/config hash
5. Add a cached-activation trainer for:
   - adapters
   - norm
   - LM head
   - generated adapters
6. Report both:
   - original-token-equivalent tokens/sec
   - cached-hidden-state steps/sec

Expected result:

- Same output path for frozen-base adapter training, because cached activations
  are the exact frozen-base outputs.
- Much lower repeated GPU compute for adapter sweeps.
- Potential 10x to 50x lower GPU training time for repeated adapter/head
  experiments on fixed data.
- Peak VRAM lower because the shared transformer graph is not built during
  adapter training.

Limitations:

- This only applies when the cached prefix is frozen.
- If module blocks or shared layers are trainable, the cache becomes stale.
- Disk use increases, so cache files should be ignored by default and only
  summary metadata should be committed.
- For large corpora, the cache should be sharded and optionally regenerated.

## Phase 4: True Sparse MoP Execution

Problem:

The Goal 46 runs report `active_param_ratio = 1.0`. This means the framework is
not yet proving sparse active-parameter execution.

Implementation:

1. Stop putting `"core"` inside `module_bank` for new MoP efficiency configs.
   The shared transformer is already the core.
2. Add `module_names` to `GPUTrainingConfig` and pass it through
   `ModelArchitectureConfig`.
3. Use module names like:
   - `["coding", "debugging", "repair"]`
4. For module-bank routing, normalize target modules with
   `always_include_core=False`.
5. Keep the shared core separate and frozen/no-grad when using sparse policies.
6. Update `estimate_active_parameters` to distinguish:
   - total active parameters
   - active trainable parameters
   - shared frozen parameters
   - routed module parameters

Why this helps:

- It removes a redundant `module:core` block.
- It makes active-module density meaningful.
- It creates a cleaner path to lower active parameter ratio without pretending
  the shared core disappears.

## Phase 4B: Expert-Heavy MoP Blocks For Real 3x To 50x Active Compute

Problem:

The current MoP design adds module-specific MLP blocks after a large shared
transformer. This is good for smoke tests, but not enough for large active
compute savings. The shared core dominates parameter count and compute.

Implemented:

1. Add an expert-heavy transformer block type:
   - shared attention
   - routed FFN experts
   - optional routed adapter/LoRA deltas
2. Move most FFN parameters into an expert bank.
3. Activate only top-1 or top-2 experts per token or per example.
4. Keep a small shared trunk for common language structure.
5. Add config fields:
   - `mop_block_type: "post_core_mlp" | "routed_ffn"`
   - `expert_count`
   - `active_experts`
   - `routing_granularity: "example" | "token"`
   - `shared_depth_ratio`
6. Add metrics:
   - `active_expert_count`
   - `expert_compute_ratio`
   - `shared_compute_ratio`
   - `estimated_active_flop_ratio`
   - `estimated_backward_flop_ratio`

Efficiency math:

- If 75% of model compute is in experts, 8 experts with top-1 routing gives
  active compute around `25% + 75% / 8 = 34.375%`, about 2.9x lower.
- If 90% of model compute is in experts, 16 experts with top-1 routing gives
  active compute around `10% + 90% / 16 = 15.625%`, about 6.4x lower.
- If 95% of model compute is in experts, 32 experts with top-1 routing gives
  active compute around `5% + 95% / 32 = 7.97%`, about 12.5x lower.
- Reaching 50x active compute requires either a tiny shared trunk, very many
  experts, cached frozen prefixes, or a narrower active path. It is not
  realistic with the current shared-core-heavy architecture.

Implemented quality guard:

Dense/post-core checkpoints are translated layer by layer. Shared layers retain
their learned weights, every routed expert receives the corresponding dense FFN
weights, and the routed block uses matching post-norm residual semantics. Tests
verify that the initial routed logits match the dense logits within numerical
precision before specialization begins.

Why this helps:

- It makes MoP efficiency structural, not only a trainable-parameter trick.
- It attacks active compute, not just optimizer memory.
- It gives a credible route to 3x to 10x active GPU savings first, then larger
  savings with more expert-heavy designs.

## Phase 4C: LoRA-Style Deltas As A Quality Safety Valve

Problem:

Post-core adapters may not recover dense-quality loss even after warm-starting,
especially on code tasks where attention and FFN internals matter.

Implemented:

1. Add LoRA-style low-rank deltas to selected linear layers:
   - attention Q/K/V projections
   - attention output projection
   - FFN up/down projections
2. Keep base weights frozen.
3. Train only low-rank deltas plus optional norm/head.
4. Route LoRA deltas by module name for MoP-specific specialization.
5. Sweep ranks:
   - `rank=4`
   - `rank=8`
   - `rank=16`

The low-rank up projections are zero-initialized, so enabling this path does
not change the warm base output before training. The first backward pass updates
the up projections while base attention and FFN weights remain frozen.

Why this helps:

- It gives much higher quality per trainable parameter than only attaching one
  adapter after the entire shared core.
- It is not quantization.
- It can stay 10x to 100x below dense trainable parameters.

Tradeoff:

LoRA-style deltas inside transformer blocks cannot use the same full
frozen-prefix activation cache, because the deltas affect intermediate hidden
states. This is a quality safety valve, not the first 50x training-time path.

## Phase 5: Train Longer, But Keep Throughput

Problem:

The Goal 46 run had only `25` optimizer steps. That is too short to judge loss,
especially for sparse training.

Implementation:

1. Add serious comparison configs with:
   - `max_steps: 2000`
   - `gradient_accumulation_steps: 8`
   - `eval_every_steps: 100`
   - `save_every_steps: 500`
   - `scheduler: "cosine"`
   - `warmup_steps: 50`
2. For adapter/norm/head runs, sweep learning rates:
   - `3e-4`
   - `1e-3`
   - `2e-3`
3. Keep dense/full baseline configs at the same token budget for fairness.
4. Add early stopping metadata but do not stop by default until the comparison
   is stable.

Important:

Longer training increases total wall-clock time, but it should not lower
tokens/sec. The target is same or better throughput per token, not fewer total
seconds for a longer experiment.

## Phase 6: Data And Evaluation Quality Gates

Problem:

Loss alone is not output quality. The report uses a tiny 50-lesson dataset.

Implementation:

1. Keep the current 50-lesson smoke path for fast regression tests.
2. Add a larger coding bugfix training dataset for the serious run.
3. Preserve a fixed held-out eval set across dense, full MoP, and sparse MoP.
4. Add generated-code evaluation after each run:
   - exact generated-code pass rate
   - verifier pass rate
   - mean eval loss
   - tokens/sec
   - peak VRAM
5. Reject any "efficient" run if output quality regresses materially.

Quality gate:

- Sparse run eval loss should be within `+0.10` to `+0.25` of dense for
  same-quality claims.
- Generated-code verifier pass rate should not fall more than 5 percentage
  points below dense.
- If sparse loss is lower but pass rate is worse, do not count it as better.

## Phase 7: Sparse Checkpoints

Problem:

Adapter-only checkpoint size is still `365.8617 MB` because the full frozen
model state is saved.

Implementation:

1. Add `save_trainable_only_checkpoints: bool = False`.
2. For sparse runs, save:
   - base checkpoint reference
   - trainable parameter state dict only
   - trainable policy
   - architecture/config hash
3. Add restore support that loads base weights first, then applies sparse delta.
4. Keep full checkpoints as the default for backward compatibility.

Expected result:

- Adapter checkpoint size should drop from hundreds of MB to single-digit MB or
  low tens of MB, depending on bottleneck and head/norm settings.
- This does not directly reduce training loss, but it is part of GPU-efficiency
  research hygiene and makes sparse MoP artifacts honest.

## Recommended First Concrete Change Set

Implement these first:

1. `resume_model_only` in GPU checkpoint resume.
2. `train_norm` support in trainable policy.
3. `adapters_norm_head` sparse policy or metadata flags for adapters plus norm
   and optional head.
4. Frozen-prefix `no_grad` execution for MoP when the shared core is frozen.
5. Config templates:
   - `100m_mop_warm_adapters_64_colab_efficiency.json`
   - `100m_mop_warm_adapters_norm_head_64_colab_efficiency.json`
   - `100m_mop_core_frozen_quality_colab_efficiency.json`
6. A comparison report template that records dense-vs-full-vs-warm-sparse.
7. Cached frozen-prefix training for adapter/norm/head sweeps.

This is the smallest set that directly attacks both problems:

- high adapter-only loss
- GPU efficiency for sparse policies

If the target is strictly 3x to 50x lower GPU usage, prioritize cached
frozen-prefix training before larger adapters. It is the only near-term feature
that can plausibly reduce repeated adapter-sweep GPU time by an order of
magnitude while keeping the same frozen-base outputs.

## Experiment Matrix

Run these in order:

| Run | Init | Trainable policy | Bottleneck | Expected role |
| --- | --- | --- | ---: | --- |
| Dense baseline extended | scratch | all | n/a | Quality reference |
| MoP full extended | scratch | all | 16 | Warm-start source |
| Warm adapter 64 | MoP full | adapters_only | 64 | Fast low-VRAM candidate |
| Warm adapter norm/head 64 | MoP full | adapters + norm + head | 64 | Quality candidate |
| Core frozen | MoP full | modules + adapters | 64 | Same-quality candidate |
| Warm adapter norm/head 128 | MoP full | adapters + norm + head | 128 | Capacity sweep |
| Cached warm adapter 64 | cached MoP full hidden states | adapters + norm/head | 64 | 10x+ GPU-time candidate |
| Routed FFN MoP | dense/MoP warm-start | top-1 experts | n/a | true active-compute candidate |

## Acceptance Criteria

Do not claim success unless a report shows:

1. Loss improves by at least 1.5x over the current MoP Adapter-Only loss:
   - eval loss `<= 3.44`
2. Stretch loss improves by 2x:
   - eval loss `<= 2.58`
3. Same-quality sparse run is close to dense:
   - eval loss within `+0.25` of dense
4. Peak reserved VRAM remains below dense:
   - target `<= 1.0 GB` for adapter/norm/head
   - target `<= 1.3 GB` for core-frozen
5. Throughput is not worse than dense:
   - target `>= 11,286 tokens/sec`
6. No quantization is used:
   - keep BF16/TF32 path
7. Output quality is checked:
   - generated-code verifier pass rate close to dense
8. Sparse checkpoint is actually sparse:
   - trainable-only checkpoint does not include the full frozen base state
9. Efficiency axis is stated honestly:
   - VRAM, trainable params, checkpoint delta, active FLOPs, and wall-clock GPU
     time are reported separately
10. A 3x to 50x claim must name the axis:
   - 3x to 5x peak VRAM is plausible near-term
   - 50x checkpoint/trainable-state reduction is plausible near-term
   - 10x to 50x training-time reduction requires cached frozen-prefix training
     or an expert-heavy routed architecture

## Why This Should Work

Adapter-only from scratch is underpowered because it freezes random weights.
Warm-starting gives the adapters a meaningful representation to adapt. Adding
LayerNorm/head training and a moderate bottleneck gives enough capacity to move
loss without training the whole model. Running the frozen prefix under
`no_grad` removes unnecessary activation bookkeeping. Sparse checkpoints make
the artifact size match the trainable parameter story.

This route does not use quantization and does not rely on lower precision than
the current BF16 run. It improves efficiency by changing what trains and what
autograd stores, not by compressing weights.

## Risks

- A larger bottleneck may reduce throughput if taken too far.
- Training the LM head can improve loss but may overfit the tiny eval split.
- Core-frozen may be the first same-quality sparse mode, while adapter-only may
  need a larger dataset or more steps.
- Current 50-example report data is too small for strong conclusions.
- Real quality must be verified with generated outputs, not only CE loss.

## Bottom Line

The next implementation should not try to make random-base adapter-only
magically good. It should make sparse MoP a proper fine-tuning mode:

1. train or load a good base,
2. freeze most of it,
3. train small but sufficient adapters/norm/head or module deltas,
4. skip autograd through frozen blocks,
5. measure loss, VRAM, speed, checkpoint size, and generated-code quality.

That is the most direct path to lower training loss and lower GPU use without
quantization or quality sacrifice.
