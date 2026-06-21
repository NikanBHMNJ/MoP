# Goal 48 - Real GPU Efficiency From Cached Sparse Training

You are working inside the MoP-Forge repository:

```text
C:\Users\GC121\Documents\mop
```

## Objective

Implement a real next-step GPU efficiency path for MoP-Forge.

The current v0.46.0 L4 warm sparse report shows good progress:

- warm adapter norm/head and warm LoRA improve short-run TinyStories loss,
- trainable parameters are much lower than Dense,
- trainable-only checkpoints are much smaller,
- throughput is higher,
- peak allocated VRAM is lower.

However, peak reserved VRAM is still too close to Dense because the sparse phase
still carries too much full-model GPU residency. The next goal is to reduce GPU
memory much more without quantization, without sacrificing speed, and without
accepting worse output quality or loss.

## Core Idea

Build a two-phase cached-tail sparse training path.

Phase 1:

- Run Dense or MoP Full normally.
- Save a warm base checkpoint.
- Generate frozen hidden-state cache records for a selected split.

Phase 2:

- Train only the sparse tail/trainable modules from cached hidden states.
- Do not keep the full frozen backbone on GPU during cached sparse training.
- Keep only the trainable adapter/norm/head/LoRA/tail modules, optimizer state,
  and current minibatch tensors on CUDA.

This is the cleanest route toward much lower GPU usage while keeping quality and
speed. The expected gain should come from removing full-backbone GPU residency
during the sparse training phase, not from quantization.

## Loss-Efficiency Idea For Code

The next reports should use a code dataset, not TinyStories. Do not compare
absolute TinyStories loss to code loss. Compare Dense, MoP Full, warm sparse,
and cached sparse on the same fixed code train/eval/test split.

Use cached sparse distillation for the code run.

Teacher:

- Train or load a stronger Dense or MoP Full code model.
- Use it as the warm teacher for sparse training.

Student:

- Start from the warm base.
- Train `cached_warm_adapter_norm_head_64` first.
- Keep the full frozen backbone off CUDA during the cached sparse phase.

Sparse student loss should combine:

1. Normal next-token cross-entropy on code.
2. Teacher KL distillation loss from Dense/MoP Full logits.
3. Optional hidden-state alignment at the cached boundary.
4. Optional hard-example replay for examples with high loss, syntax failures, or
   verifier failures.

Prefer top-k teacher logits for distillation cache records so the cache does not
become a huge model artifact. Keep all large caches and checkpoints out of Git.

The goal is not only lower final loss. The goal is lower loss per GPU memory and
lower time-to-target-loss.

## Training Loss Efficiency Plan

Improve loss efficiency without hiding the cost or lowering the benchmark
difficulty.

Use a staged code-training path:

1. Train a Dense or MoP Full teacher on the exact same fixed code split.
2. Save the best warm checkpoint by eval loss, not only by final train step.
3. Generate cached hidden states from the frozen warm base.
4. Store top-k teacher logits with each cached record.
5. Train cached sparse students with CE plus top-k KL distillation.
6. Replay hard examples that remain high-loss or fail syntax/verifier checks.
7. Run a short uncached polish pass only if cached sparse loss is close but
   generated-code quality still lags.

The preferred first student remains:

```text
cached_warm_adapter_norm_head_64
```

If it underfits, try these in order:

```text
cached_warm_adapter_norm_head_128
cached_warm_lora_rank8
cached_warm_lora_rank16
```

Do not claim a loss win from easier data, shorter context, fewer eval examples,
or a different split. Dense, MoP Full, warm sparse, and cached sparse must be
measured on the same train/eval/test split with the same tokenizer, sequence
length, batch policy, and eval cadence.

Loss-efficiency reports should include:

- best eval loss,
- final eval loss,
- time-to-target-loss,
- tokens-to-target-loss,
- peak allocated VRAM at target loss,
- peak reserved VRAM at target loss,
- trainable parameter ratio,
- checkpoint size,
- syntax/compile pass rate when available,
- verifier or task pass rate when labels support it.

The target is not just lower loss. The target is comparable or better loss at a
much lower trainable-state and GPU-memory cost.

## Output Quality Solution

The Goal 48 code report proves the cached sparse efficiency path, but it also
shows the next bottleneck clearly: small models do not yet generate useful code
reliably. The next solution is to improve output quality while preserving:

- speed,
- GPU efficiency,
- train loss and eval loss,
- trainable-only checkpoint size,
- sparse/cached measurement discipline.

Do not solve quality by abandoning the efficiency path. Keep cached sparse
training, frozen-backbone offload, trainable-only checkpoints, teacher top-k
distillation, and lightweight reports.

Repository implementation hook:

```text
mopforge gpu prepare-efficiency-data --quality-format fixed_code_xml
```

Implemented quality experiment notebook:

```text
notebooks/colab_l4_goal49_verified_code_quality_report.ipynb
```

The notebook uses cache-compatible `lora_tail_only=true` for cached LoRA rank
8/16. This keeps LoRA deltas after the cached hidden-state boundary. Cached
runs restore the full model only after training for generated-code evaluation,
write samples to `generation_eval.json`, and keep that evaluation outside the
cached-tail VRAM measurement window.

This frames verified code-repair targets as:

```text
<fixed_code>...</fixed_code>
```

Generation evaluation must extract this block before running syntax/verifier
checks. Keep `raw` as the default data format for backward-compatible reports.

Use two compatible quality tracks.

### Track A - Small Efficient Specialist Models

Small models should be trained as specialist students, not as broad free-form
code generators.

Preferred task framing:

```text
code repair
patch generation
function completion
test-error-conditioned fixing
short verified snippets
```

Avoid asking 100M-scale models to generate arbitrary programs from scratch. For
small models, require narrow outputs such as:

```text
<fixed_code>...</fixed_code>
```

or a compact patch/diff format.

Training recipe:

1. Train or load a stronger Dense, MoP Full, or larger teacher on the same fixed
   code split.
2. Generate teacher candidate repairs/completions.
3. Run syntax checks and task verifiers/tests.
4. Keep only teacher outputs that pass verification as quality targets.
5. Cache hidden states plus teacher top-k logits and teacher CE loss.
6. Train cached sparse students with:
   - normal next-token CE,
   - teacher top-k KL,
   - supervised CE on verified teacher outputs,
   - hard-example replay for high-loss or failing examples,
   - optional preference/ranking loss where passing teacher output beats failing
     student output.
7. Run verifier-guided data refresh: sample from the student, verify outputs,
   add failures to replay data, and repeat.

Preferred small-student profiles:

```text
cached_warm_adapter_norm_head_128
cached_warm_lora_rank8
cached_warm_lora_rank16
```

Keep `cached_warm_adapter_norm_head_64` as the efficiency baseline, but use 128
or LoRA rank 8/16 when output quality is the primary blocker.

Acceptance for small models:

- syntax pass rate improves,
- verifier pass rate improves,
- exact match or task success improves,
- eval loss remains close to the warm sparse baseline,
- tokens/sec remains better than or close to Dense,
- peak reserved VRAM remains much lower in the cached-tail phase,
- trainable ratio and checkpoint size remain sparse.

### Track B - Larger And Frontier-Scale MoP Models

MoP-Forge should not be framed as only a tiny-model trick. The same framework
must support larger and frontier-scale training profiles.

For larger models, the quality path can include broader code generation because
the model has enough capacity to learn syntax, long-range structure, and task
intent.

Scale targets:

```text
small: 100M to 1B efficient specialist students
medium: 1B to 7B MoP specialist or general-code models
large/frontier: multi-expert MoP with longer context and distributed training
```

Shared framework requirements across scales:

- one data/report schema for lessons, code repair, completions, tests, verifier
  metadata, and generated samples,
- one comparison schema for Dense, MoP Full, warm sparse, cached sparse, LoRA,
  adapter/norm/head, and routed experts,
- one evidence standard for loss, throughput, VRAM, checkpoint size, syntax
  pass, verifier pass, exact match, and time-to-target-loss,
- no unsupported quality claims without measured reports.

Large/frontier recipe:

1. Train a stronger Dense or MoP Full teacher/generator on broader code data.
2. Add routed MoP experts for code domains such as repair, debugging, tests,
   APIs, algorithmic snippets, and refactoring.
3. Use self-distillation or teacher distillation into sparse MoP profiles.
4. Use generated-code verification and replay loops at scale.
5. Measure not only final loss, but quality per GPU memory and time-to-target
   quality.

This gives MoP-Forge a single story:

```text
Small models become efficient verified specialists.
Large models become broader code generators with sparse MoP specialization.
Both use the same training, caching, verification, and reporting framework.
```

### Goal 49 Quality Experiment - Completed

Run the next code report with:

```text
Dense code teacher
MoP Full code teacher
Warm Adapter Norm/Head 128
Cached Warm Adapter Norm/Head 128 + teacher top-k KL
Cached Warm LoRA rank8 + teacher top-k KL
Optional Cached Warm LoRA rank16 + teacher top-k KL
```

Use a fixed code-repair dataset with verifier tests. Configure
`target_eval_loss` after the baseline run. Add verified teacher targets and
hard-example replay before claiming output-quality recovery.

Primary expected result:

```text
cached sparse should preserve the speed, VRAM, and checkpoint advantages while
closing the syntax/verifier quality gap on narrow code repair tasks.
```

Do not claim that a small cached sparse student is a general code generator
unless generated samples, syntax pass rate, verifier pass rate, and task success
prove it.

Measured Goal 49 outcome:

- cached Adapter/Norm/Head 128 and cached tail-only LoRA Rank 8 preserved the
  major throughput, VRAM, trainable-ratio, and checkpoint advantages,
- both cached profiles reached 50% syntax pass on 32 generated samples,
- exact match and verifier pass remained 0% for every profile,
- the report therefore validates cached-tail efficiency, not useful generated
  code quality.

### Goal 50 - Evaluation Integrity And 100M Learning Gate

Do not scale this experiment to 1B yet. First prove that the 100M pipeline can
learn and generate executable solutions for the narrow verified repair tasks.

The Goal 49 audit found:

- 1,000 total lessons: 800 train, 100 eval, and 100 test,
- 1,500 microsteps produced only 188 optimizer updates,
- cached hard replay expanded the train loader to 1,439 records, so the cached
  runs completed only about one replay-expanded pass,
- all profiles evaluated loss on only two examples because `eval_batches=2`,
- generation used the final in-memory weights instead of restoring the best
  eval-loss checkpoint,
- training used `shuffle=False` and loaded records in five category-grouped
  blocks,
- the first 32 generation examples covered only missing-return and off-by-one
  tasks, not all five categories,
- all 1,000 prompt/target pairs fit within sequence length 1,024 and all targets
  fit within the 256-token generation budget,
- all 100 held-out ground-truth solutions passed exact match and the verifier in
  both raw and `<fixed_code>` framing.

This means the extraction/verifier core is functioning. The primary blockers
are evaluation integrity, too few effective optimizer updates, category-ordered
training, limited data diversity, and autoregressive byte-level generation.
Model capacity remains a hypothesis, not the first conclusion.

#### Phase A - Correct The Experiment Protocol

Before another quality report:

1. Shuffle the train loader every epoch with a recorded deterministic seed.
2. Preserve fixed train/eval/test membership while avoiding source-order
   category blocks.
3. Evaluate loss over all 100 held-out eval lessons, or record an explicitly
   justified larger eval subset. Never derive best loss from two examples.
4. Restore the best eval-loss checkpoint before generated-code evaluation.
5. Record the exact checkpoint path and step used for every quality result.
6. Evaluate all 100 held-out lessons for the diagnostic run. If a subset is
   required later, stratify it across all five bug categories.
7. Report per-category counts and metrics for XML completion, syntax, exact
   match, verifier pass, and failure type.
8. Keep ground-truth raw/XML verifier controls in every report. These controls
   must remain 100% passing.
9. Report both microsteps and optimizer updates prominently.
10. Keep generation limits and truncation statistics in experiment settings so
    a target can never be silently truncated.

Implemented repository support:

- `GPUTrainer` now advances real DataLoader epochs instead of cycling one fixed
  iterator, and records epoch, shuffle policy, and shuffle seed,
- quality configs can evaluate the full held-out loader,
- generated-code evaluation can restore and record the best eval checkpoint,
  evaluate train and held-out splits, and select a deterministic stratified
  subset,
- generation artifacts include complete-XML and per-category quality metrics,
- every GPU quality evaluation writes raw/XML ground-truth controls,
- lesson data metadata records pre-truncation prompt, target, and sequence
  lengths,
- fixed dataset splits can stratify on `bug_type`,
- `notebooks/colab_l4_goal50_100m_learning_gate.ipynb` implements the Phase B
  L4 diagnostic and report gate.
- `notebooks/colab_l4_goal50_100m_quality_comparison.ipynb` implements the gated
  Phase C Dense/MoP/warm/cached 100M comparison and writes explicit
  quality-plus-efficiency acceptance gates.

#### Phase B - 100M Memorization Diagnostic

Run a deliberately small learning test before another full comparison:

- 50 verified lessons total,
- 10 lessons from each of the five bug categories,
- a separate balanced held-out set,
- seeded shuffling,
- 1,000 to 2,000 optimizer updates, not microsteps,
- full train and held-out generation from the best checkpoint,
- no quantization.

Memorization acceptance gates:

- ground-truth verifier control: 100%,
- complete `<fixed_code>` output rate: at least 95% on train,
- train syntax pass: at least 95%,
- train verifier pass: at least 95%,
- train exact match: at least 90%,
- metrics include all five categories and the checkpoint used.

Interpretation rules:

- If train verifier remains low, the training/generation pipeline is still
  defective or materially undertrained. Do not blame data generalization or
  model size.
- If train verifier passes but held-out verifier remains low, expand and
  diversify verified training data.
- If train and held-out quality improve but plateau after protocol and data
  fixes, model capacity becomes a credible blocker.

Measured first diagnostic outcome:

- protocol, full-eval, category-coverage, best-checkpoint, ground-truth-control,
  truncation, and optimizer-update checks passed,
- best eval loss reached `0.0129`,
- train and held-out XML completion, syntax, verifier, and exact match were all
  `0%`,
- therefore the memorization gate failed and Phase C/1B remain blocked,
- the next investigation must focus on the teacher-forced-loss versus
  autoregressive-generation mismatch before adding model scale or data volume.

Post-report root cause and fix:

- supervised training used `BOS + prompt + target + EOS`,
- greedy generation incorrectly began from `BOS + prompt + EOS`,
- generation now begins from `BOS + prompt`, matching the trained target
  boundary,
- the failed report remains available in Git history as pre-fix evidence.

Measured corrected rerun outcome:

- all protocol and ground-truth controls passed again,
- train and held-out XML completion, syntax, verifier, and exact match reached
  `100%`,
- best eval loss reached `0.0000904`,
- the memorization gate now passes and Phase C is allowed,
- 1B remains blocked until the full 100M comparison passes its quality and
  efficiency gates.

#### Phase C - Full 100M Quality Comparison

Proceed only after the memorization gate passes.

Use:

- at least 10,000 verified repair/completion lessons,
- balanced categories and a fixed leakage-checked split,
- at least 2,000 optimizer updates,
- seeded epoch shuffling,
- full held-out loss evaluation,
- early stopping and generation from the best checkpoint,
- Dense, MoP Full, Warm Adapter/Norm/Head 128, Cached Adapter/Norm/Head 128,
  and Cached Tail-Only LoRA Rank 8,
- the same tokenizer, sequence length, batch policy, eval cadence, and
  generation budget for every profile.

Minimum quality gate before scaling:

- ground-truth verifier controls remain 100%,
- complete output framing is at least 90%,
- held-out syntax pass is at least 80%,
- held-out verifier pass is at least 20% and materially better than Goal 49,
- exact match is nonzero,
- cached profiles retain a measured efficiency advantage,
- no broad code-generation claim is made from these narrow repair tasks.

Measured Phase C outcome:

- all five profiles completed 2,000 optimizer updates on the same balanced
  10,000-lesson fixed split with full held-out loss and best-checkpoint
  generation,
- Cached Adapter/Norm/Head 128 reached `88.0%` verifier/exact match versus
  Dense at `82.4%`, with `8.35x` throughput, `31.70x` lower peak reserved VRAM,
  and a `127.53x` smaller checkpoint,
- Cached Tail-Only LoRA Rank 8 also reached `88.0%` verifier/exact match, with
  `6.70x` throughput and `22.83x` lower peak reserved VRAM,
- all ground-truth, category, best-checkpoint, full-eval, optimizer-budget,
  framing, syntax, verifier, exact-match, throughput, and VRAM gates passed,
- the generated report initially false-failed because cached loaders omit
  sequence-length metadata; shared fixed-split evidence records zero truncation
  and a 166-token maximum target within the 256-token budget,
- correcting that report-only fallback changed no measured run value,
- the run used a fixed update budget rather than early stopping,
- Phase C supports a 1B memory/throughput probe, not a broad code-generation
  claim or an automatic 1B full-training claim.

#### 1B Scaling Gate

A 1B run may begin only after:

1. the 100M memorization diagnostic passes,
2. the full 100M run demonstrates nonzero held-out verifier and exact-match
   quality,
3. evaluation uses the best checkpoint and all categories,
4. the larger dataset and split are reproducible,
5. a short 1B L4 memory/throughput probe stays within the measured VRAM budget.

Until those gates pass, a 1B run would make the same unresolved experiment more
expensive rather than proving model-scale quality.

## Target Profiles

Optimize this path first:

```text
warm_adapter_norm_head_64
```

Then compare:

```text
dense
mop_full
warm_adapter_norm_head_64
warm_lora_rank8
cached_warm_adapter_norm_head_64
cached_warm_lora_rank8
```

The warm adapter norm/head profile is currently the strongest candidate because
the v0.46.0 L4 TinyStories report showed it had:

- lower eval loss than Dense in the short run,
- much higher tokens/sec,
- far fewer trainable parameters,
- a tiny trainable-only checkpoint.

## Implementation Requirements

Add a cached sparse training mode that can:

1. Load a frozen base checkpoint for cache generation.
2. Run the frozen prefix/backbone once over the corpus.
3. Write lightweight hidden-state cache shards with metadata.
4. Train sparse tail modules from those cached hidden states.
5. Avoid loading the full frozen backbone onto CUDA during cached-tail training.
6. Save trainable-only checkpoints that reference the base checkpoint and cache
   manifest.
7. Report allocated VRAM, reserved VRAM, final reserved VRAM, tokens/sec,
   samples/sec, train loss, eval loss, trainable ratio, checkpoint size, and
   cache metadata.
8. Keep all model artifacts/checkpoints out of Git.

Do not use quantization as the main efficiency mechanism.

## Measurement Rules

GPU efficiency claims must name the exact axis improved:

- peak allocated VRAM,
- peak reserved VRAM,
- final reserved VRAM after cache cleanup,
- throughput,
- trainable parameter ratio,
- active trainable ratio,
- checkpoint size,
- eval loss,
- generated-code quality when a code dataset is used.

Report both allocated and reserved VRAM. Reserved VRAM can be inflated by the
PyTorch caching allocator, so also measure phase boundaries after cleanup with:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

Do not overclaim. A sparse run is a same-quality efficiency win only if it stays
close to Dense eval loss and improves a named efficiency axis.

## Proposed Experiment

Use Google Colab L4 first.

Initial workflow-validation dataset:

```text
roneneldan/TinyStories
```

Next evidence dataset:

```text
fixed code dataset / code-repair dataset
```

Preferred code-data path:

- use the repo's fixed-split coding lesson or bugfix dataset tools if they are
  already available,
- otherwise use a CodeAlpaca-style corpus converted into the same MoP-Forge
  corpus format,
- preserve train/eval/test split metadata in every report.

Recommended run:

1. Dense warmup: 300 to 1000 steps.
2. MoP Full warmup: 300 to 1000 steps.
3. Cache frozen hidden states once.
4. Train `cached_warm_adapter_norm_head_64` for 1000 to 3000 steps.
5. Train `cached_warm_lora_rank8` for 1000 to 3000 steps.
6. Compare against non-cached Dense, MoP Full, warm adapter, and warm LoRA.

For the code report, add:

1. Dense code baseline.
2. MoP Full code teacher.
3. Warm Adapter Norm/Head 64 code run.
4. Cached Warm Adapter Norm/Head 64 with teacher top-k KL.
5. Optional Cached Warm LoRA Rank 8 with teacher top-k KL.
6. A short end-to-end polish pass if cached sparse loss is close but generated
   code quality needs recovery.

Primary expected result:

```text
cached_warm_adapter_norm_head_64 should keep loss close to the warm adapter
profile while reducing GPU memory much more than the current non-cached sparse
path.
```

## Acceptance Gates

A cached sparse profile is promising only if:

- eval loss is close to Dense or better on the same split,
- tokens/sec is not worse than Dense,
- peak allocated VRAM is substantially lower than Dense,
- peak reserved VRAM is lower after phase cleanup,
- trainable parameter ratio remains much lower than Dense,
- checkpoint size remains much smaller than Dense,
- reports contain enough metadata to reproduce the comparison.

For code reports, also require:

- exact match or task success where labels support it,
- verifier pass rate,
- syntax/compile pass rate when available,
- generated-code sample artifacts,
- time-to-target-loss compared with Dense and MoP Full.

Stretch target:

```text
3x to 50x lower trainable-state/checkpoint footprint and meaningfully lower
GPU memory, while keeping speed and loss competitive.
```

Do not claim 3x to 50x lower full end-to-end GPU memory unless the measured
reserved and allocated VRAM actually prove it.

## Files To Consider

Inspect the existing implementation before editing:

```text
mopforge/gpu/
mopforge/models/
mopforge/pretrain.py
configs/jobs/
notebooks/colab_l4_v046_efficiency_comparison.ipynb
docs/warm_sparse_efficiency_comparison_template.md
reports/v0_46_0_l4_warm_sparse_comparison/
tests/
```

Prefer extending existing GPU trainer/config/report patterns instead of adding a
parallel framework.

## Docs And Reports

When implemented, update:

```text
README.md
docs/README.md
docs/gpu_quickstart.md
docs/warm_sparse_efficiency_comparison_template.md
notebooks/colab_l4_v046_efficiency_comparison.ipynb
```

If a new experiment is run, add a lightweight report under:

```text
reports/
```

The report must not include:

```text
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
```

## Validation

Before committing implementation changes, run:

```powershell
python -m pytest -q
python scripts/release_check.py --quick-examples
```

For notebook/report-only changes, also validate:

```powershell
python -m json.tool notebooks/colab_l4_v046_efficiency_comparison.ipynb
git diff --check
git diff --cached --name-only
```

Make sure no checkpoint or model-weight files are staged.

## Final Response Requirements

When finished, report:

1. what was implemented,
2. which files changed,
3. what GPU-efficiency axis the change targets,
4. what tests/checks passed,
5. whether any checkpoint/model files were detected,
6. final `git status`.
