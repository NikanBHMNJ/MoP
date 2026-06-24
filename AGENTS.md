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

### Goal 51 - A100 1B Admission And Feasibility

The Goal 50 memorization and full 100M comparison gates now pass. The next step
is not an immediate full 1B training run. First make MoP-Forge scale-aware and
run a staged 1B admission probe on A100 40 GB and 80 GB hardware.

Do not promise that a hardware run or model-quality target is guaranteed.
Instead, guarantee that the workflow stops before expensive training unless
every measurable prerequisite passes.

#### Framework Prerequisites

Implement these before the first 1B pilot:

1. Add an explicit `max_optimizer_steps` budget while preserving backward
   compatibility with `max_steps` microsteps.
2. Drive warmup, cosine decay, evaluation cadence, checkpoint cadence, and
   reporting from clearly named optimizer-step or microstep units. The current
   scheduler advances per optimizer update but derives its duration from
   `max_steps`; this must not remain ambiguous at 1B scale.
3. Add a staged command such as:

   ```text
   mopforge gpu probe <config>
   ```

   It must separately measure model allocation, forward, backward, optimizer
   initialization/step, evaluation, cleanup, checkpoint save, and checkpoint
   reload.
4. Make the memory estimator reflect actual runtime storage. BF16 autocast does
   not imply that model parameters and optimizer state occupy BF16 storage; the
   estimate must inspect or explicitly model parameter, gradient, optimizer,
   master-weight, activation, and transient workspace dtypes.
5. Record at every probe phase:
   - allocated and reserved CUDA memory,
   - peak allocated and reserved CUDA memory,
   - free and total device memory,
   - allocator retries and OOM counters when available,
   - non-releasable/cached memory where available,
   - host RAM and report/checkpoint disk requirements,
   - phase duration and tokens/sec.
6. Catch CUDA OOM during a probe, write a failure report, clean up CUDA state,
   and stop. Do not silently lower sequence length, batch size, model size, or
   benchmark difficulty.
7. Propagate source sequence-length, truncation, split, and tokenizer metadata
   into activation-cache manifests and cached loaders. Shared dataset evidence
   must not require a report-time fallback.
8. Add atomic model-only checkpoint writes and a mandatory save/reload/resume
   probe. Probe checkpoints should omit optimizer state unless resume testing
   explicitly requires it.
9. Add incremental KV-cached generation before large 1B generation
   evaluations. The current greedy path recomputes the complete context for
   every generated token and is not acceptable for a full 1B quality report.
10. Keep quantization, FP8, CPU optimizer offload, FSDP, and DeepSpeed outside
    the first single-A100 comparison unless a later measured failure justifies a
    separately reported experiment.

#### A100 Admission Profiles

Create executable, not plan-only, profiles for both hardware classes.

Common conservative starting policy:

```text
precision: bf16 autocast
allow_tf32: true
max_seq_len: 1024
micro_batch_size: 1
gradient_accumulation_steps: 16 or 32
activation_checkpointing: true
efficient_attention: torch_sdpa
compile_model: false for the first probe
generation evaluation: disabled for allocation/backward probes
optimizer checkpoint state: disabled for the first probe
quantization: none
```

Keep sequence length at 1,024 for the first comparison because the measured
Goal 50 dataset fits within it. Do not spend A100 memory on a 2,048-token context
until a new dataset and quality objective require it.

A100 40 GB admission gate:

- peak reserved VRAM should remain at or below approximately 34 GB,
- leave enough headroom for allocator and checkpoint transients,
- do not increase microbatch size above one until the complete optimizer-step
  probe passes.

A100 80 GB admission gate:

- begin with the same conservative profile used on 40 GB,
- peak reserved VRAM should remain at or below approximately 68 GB,
- use the additional memory only after the common profile passes, so 40 GB and
  80 GB evidence remains comparable.

Proposed tracked configs:

```text
configs/jobs/1b_dense_a100_40gb_probe.json
configs/jobs/1b_mop_full_a100_40gb_probe.json
configs/jobs/1b_cached_adapter_128_a100_40gb_probe.json
configs/jobs/1b_dense_a100_80gb_probe.json
configs/jobs/1b_mop_full_a100_80gb_probe.json
configs/jobs/1b_cached_adapter_128_a100_80gb_probe.json
```

Proposed notebook:

```text
notebooks/colab_a100_goal51_1b_feasibility_probe.ipynb
```

The notebook must detect total A100 memory and select only the matching 40 GB or
80 GB profile. It must never assume that an A100 label implies a specific memory
capacity.

#### Probe Stages

Run each profile through these hard gates:

1. Validate config, dataset, tokenizer, model shape, and real parameter count.
2. Produce a static memory estimate with a named safety margin.
3. Allocate the model and record baseline CUDA/host memory.
4. Run forward-only warmup and measured steps.
5. Run backward without an optimizer step.
6. Initialize AdamW, run 20 to 50 optimizer updates, and verify finite,
   decreasing loss.
7. Reset peaks at phase boundaries and verify cleanup behavior.
8. Save a lightweight/model-only checkpoint, reload it, resume, and compare the
   resumed loss to the uninterrupted path.
9. Project 500-update and 2,000-update runtime from measured steady-state
   throughput.
10. Write a lightweight admission report with no weights or optimizer state.

The probe passes only if:

- no phase OOMs,
- every loss is finite,
- optimizer updates actually equal the requested count,
- reserved VRAM remains below the hardware-specific gate,
- checkpoint save/reload/resume succeeds,
- cleanup and final reserved memory are reported,
- measured throughput supports the requested run within the available hardware
  window,
- no benchmark, context, batch, or model setting is silently changed.

#### 1B Pilot After Admission

After the matching A100 admission report passes, run a 500-optimizer-update
pilot with only:

```text
Dense 1B
MoP Full 1B teacher/warm base
Cached Adapter/Norm/Head 128 1B student
```

Cached Adapter/Norm/Head 128 is the primary 1B student because Goal 50 measured
the strongest combined quality, throughput, VRAM, time-to-target, and
checkpoint result for that profile. Keep Cached Tail-Only LoRA Rank 8 as a
secondary follow-up, not a required first pilot.

Pilot requirements:

- use the same tokenizer, sequence length, fixed split, optimizer-update budget,
  eval cadence, and predeclared target loss across comparable profiles,
- keep full held-out loss evaluation,
- use a deterministic stratified generation subset of at least 50 examples,
- generate from the best checkpoint,
- retain ground-truth controls and all five bug categories,
- use a more diverse, leakage-audited code-repair evaluation set in addition to
  the templated Goal 50 benchmark,
- report quality and efficiency separately,
- exclude all model/checkpoint/cache artifacts from Git.

Only proceed to a 2,000-optimizer-update 1B comparison if the pilot passes its
memory, resume, loss, syntax, verifier, exact-match, throughput, and
time-to-target gates. Require a second seed before any strong research claim.

Goal 51 evidence supports only the measured A100 hardware class, memory size,
dataset, model profile, context, seed, and training budget. A 40 GB pass does
not automatically prove an 80 GB result, and an 80 GB pass must not be used to
claim 40 GB feasibility.

### Goal 52 - Production Decoder And 2B H100 Readiness

Goal 51 and the narrow Goal 50 quality gates are prerequisites, not a complete
path to a usable 2B model. Goal 52 makes the implementation scale-aware before
spending a long H100 allocation.

Implemented production foundation:

1. `production_decoder_v2` uses RoPE, RMSNorm, grouped-query attention, SwiGLU,
   PyTorch SDPA, activation checkpointing, and native incremental K/V caching.
2. The same decoder supports Dense, oracle MoP, and learned top-k token-routed
   MoP feed-forward execution.
3. `mopforge tokenizer train-bpe` trains a local byte-level BPE tokenizer and
   records source hashes and immutable special-token IDs.
4. `mopforge gpu pack-corpus` creates deterministic, fixed-length,
   memory-mapped token shards with document-level train/eval membership and
   shard hashes.
5. `GPUTrainer` supports torchrun DDP/FSDP, rank-aware samplers, no-sync
   accumulation, optimizer-step/token budgets, exact data-cursor resume, and
   deterministic process-group cleanup.
6. Distributed Checkpoint shards preserve model and optimizer state.
   `mopforge gpu consolidate-checkpoint` reconstructs a model-only checkpoint
   for evaluation, post-training, and export.
7. Production post-training includes verified SFT through `GPUTrainer`,
   one-time reference-logprob-cached DPO, and reference-free ORPO.
8. Standard code evaluation accepts trusted HumanEval-, MBPP-, or native-style
   JSONL and records pass@1, syntax, exact match, task failures, and a separate
   contamination audit against named training sources.
9. `mopforge model export-hf` writes Llama-compatible config, tokenizer, and
   sharded weights for a Dense model or one explicitly materialized MoP expert.

Tracked H100 admission sizes:

```text
304M Dense: 304,137,216 parameters
1B Dense: 1,015,779,072 parameters
2B Dense: 2,082,246,912 parameters
2B-class routed MoP: 2,480,265,984 total, about 1,015,779,072 active
```

Run `notebooks/colab_h100_goal52_2b_readiness.ipynb`. It must detect an H100
80 GB or 94 GB memory tier, build or validate the tokenizer and packed shard
manifests, and require passing 304M and 1B reports before starting the matching
2B probe. Do not silently alter model dimensions, context, microbatch, or
accumulation after failure.

After single-H100 admission, the first distributed pilots are:

```text
configs/jobs/goal52_2b_dense_8xh100_fsdp_pilot.json
configs/jobs/goal52_2b_mop_8xh100_fsdp_pilot.json
configs/jobs/goal52_2b_verified_sft_8xh100_fsdp.json
```

Each pilot is a 500-optimizer-update gate, not a full pretraining claim. Require
sharded save/reload/resume, finite loss, exact update/token counts, per-rank
VRAM, throughput, host/disk telemetry, and no automatic workload change. Keep
checkpoints, optimizer state, corpora, tokenizer artifacts, and token shards
out of Git.

Before calling any model usable, require held-out loss/perplexity, standard code
pass@1, contamination evidence, verified repair pass/syntax rates, generated
samples and failure categories, checkpoint-resume equivalence, named hardware
efficiency, and a second seed for strong comparative claims. The local Python
verifier is not a secure sandbox and may run only trusted code.

Goal 52 implementation evidence and commands are documented in:

```text
docs/production_2b_readiness.md
reports/goal52_h100_2b_readiness/
```

The report directory currently contains a schema only. Do not claim measured
H100 feasibility or quality until the generated hardware reports are added.

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
