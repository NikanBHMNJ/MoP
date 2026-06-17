# MoP-Forge

**Version:** `0.46.0`

MoP-Forge is a local-first research framework for testing
Mixture-of-Parameters training ideas. The current release is a v1.0-beta
release candidate: it has stable local data/model/run/benchmark/report
plumbing, CPU-safe smoke tests, and a serious single-GPU research beta for
tiny-to-small MoP experiments plus validated large-job profiles. It is not a
production distributed LLM training framework, and it has not yet demonstrated
large-scale MoP superiority.

## Quick Start

```bash
pip install -e .[dev]
python -m pytest -q
mopforge version
mopforge doctor
```

CPU smoke path:

```bash
python examples/create_lessons.py
python examples/run_tiny_trainer.py
python examples/run_benchmarks.py
python examples/analyze_results.py
```

GPU beta planning path:

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu launch-torchrun configs/jobs/multigpu_mop_torchrun_plan.json --dry-run
```

On CPU-only machines, tiny GPU jobs can run as CPU fallback smoke tests when
their config allows fallback. That does not validate GPU throughput or memory
behavior. Serious A100/H100 experiments require user-provided hardware, local
data, and explicit execution.

## What Is Truly Implemented

- A structured Knowledge Training Store with JSONL lessons, validation,
  filtering, SQLite metadata indexing, feedback records, curriculum scheduling,
  and module training queue metadata.
- Tiny dense, oracle-routed MoP, learned-router, fast-adapter, and generated
  parameter smoke-model paths for local PyTorch experiments.
- CPU-first TinyTrainer, SFT, continued-pretraining, checkpoint/resume,
  artifact registry, experiment runner, benchmark suite, analysis reports,
  statistics tables, dataset registry, model registry, run manifests, result
  importer, ablations, baselines, and Markdown paper report scaffolds.
- Runtime/device planning for CPU, CUDA, MPS, precision policy, autocast, model
  and batch movement helpers, and runtime metadata.
- A serious single-GPU beta package under `mopforge.gpu`: `GPUTrainingConfig`,
  `GPUTrainer`, AMP/GradScaler behavior, gradient accumulation, GPU data
  loading, checkpoint/resume, memory estimates, A100/H100 job profiles,
  torchrun dry-run planning, and MoP/Fast-Parameter routing metadata.
- GPU-efficiency benchmarking support: nested efficiency metrics for GPU runs,
  CUDA peak/current memory tracking, sparse MoP trainable-policy modes,
  Colab-safe dense/MoP comparison configs, and JSON/CSV run comparison tooling.
- Release-polish utilities: curated public API policy, `mopforge doctor`,
  smoke-example runner, release-check script, docs index, command cookbook,
  release notes, and v0.46.0 hardening tests.

## Command Overview

```bash
mopforge version
mopforge doctor
mopforge config write-default gpu_tiny_smoke outputs/gpu_tiny_smoke.json
mopforge config validate configs/examples/runtime_cpu.json
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
mopforge gpu resume <checkpoint_or_run_id>
mopforge gpu list
mopforge gpu show <run_id>
```

## Documentation Index

- [Docs home](docs/README.md)
- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Public API overview](docs/api_overview.md)
- [Config templates](docs/config_templates.md)
- [Examples guide](docs/examples_guide.md)
- [GPU quickstart](docs/gpu_quickstart.md)
- [Colab 100M training notebook](docs/colab_training.md)
- [GPU job profiles](docs/gpu_job_profiles.md)
- [GPU efficiency benchmarking](docs/gpu_efficiency_benchmarking.md)
- [GPU/runtime limitations](docs/gpu_runtime_limitations.md)
- [Serious jobs checklist](docs/serious_jobs_checklist.md)
- [Command cookbook](docs/command_cookbook.md)
- [Known limitations](docs/known_limitations.md)
- [Research positioning](docs/research_positioning.md)
- [Release checklist](docs/release_checklist.md)

## v1.0-beta Path

The next release step is evidence, not scope expansion: run tiny GPU smoke,
100M dense, 100M MoP, dense-vs-MoP benchmarks, then 500M H100 profiles before
attempting 1B/2B planning profiles. Production FSDP/DeepSpeed, custom CUDA
kernels, distributed training hardening, and remote/cloud execution remain out
of scope for this release candidate.

## Safety Warnings

- `mopforge.verify.verify_python_solution` runs local Python code and is not a
  secure sandbox.
- CPU remains the default and must stay stable.
- FP8 is planning-only.
- Torchrun support is a dry-run launcher foundation.
- MoP routing and Fast Parameters are PyTorch-level experimental paths, not
  fused-kernel production implementations.
- In plain terms, MoP-Forge is not a production distributed LLM training framework.

MoP-Forge is an experimental Python framework for future Mixture-of-Parameters
training research on modest hardware. This repository currently implements the
Knowledge Training Store, a verified Python coding/debugging lesson builder,
and a small model-ready causal-LM data pipeline with a tiny dense baseline for
smoke tests. It now also includes a tiny oracle-routed Mixture-of-Parameters
causal LM prototype and a tiny learned module router for CPU smoke tests. It
also includes a generated-code evaluation loop that extracts model output and
checks it with lesson tests, plus a repair-loop MVP that turns failed
generations into new structured lessons. The KTS now keeps JSONL as the
canonical record store while adding a SQLite metadata index for fast local
queries. It also includes a deterministic curriculum scheduler over the indexed
KTS and a tiny curriculum-driven run loop with a file-backed run registry. It
now adds an offline feedback-aware curriculum layer that turns generated-code
evaluation outcomes into per-lesson feedback summaries and prioritized
curriculum plans. It also includes the first tiny closed feedback loop that
uses feedback-weighted curriculum to retrain, re-evaluate generated code, and
append new feedback records. It now has a local SQLite module-specific
training queue layer for CPU-smoke scheduling metadata, plus a local artifact
manifest and tiny checkpoint manager. It now includes a reusable CPU-first tiny
trainer skeleton that unifies config, curriculum, model setup, train/eval,
checkpoints, artifacts, and run records. It now includes module-specific
trainable-parameter policies that can freeze or unfreeze tiny model parameter
groups for CPU-smoke trainer runs. It now includes a first fast-adapter MVP:
small named adapter banks that can be selected from lesson target modules and
trained through the policy system. It now exposes FT/SFT training modes as a
first-class API that maps supervised objectives into TinyTrainer configs,
lesson filters, parameter policies, and run metadata. It does not include a
production distributed training stack yet. It now adds a
continued-pretraining corpus API for raw/semi-structured text/code streams,
full-sequence causal-LM chunks, and
tiny CPT run artifacts. It now adds a tokenizer abstraction layer with
serializable tokenizer specs, a tokenizer factory, preserved byte-tokenizer
defaults, optional local Hugging Face/tokenizers compatibility, and tokenizer
metadata artifacts for tiny SFT/CPT runs. It now adds the first generated
parameter MVP: a tiny hypernetwork-generated adapter path conditioned on module
names, distinct from normal fast adapters and still CPU-smoke only. It now has
a lightweight config and CLI layer for JSON/YAML config envelopes, CPU-safe
default templates, validation/dry-runs, and local SFT/CPT/TinyTrainer runs. It
now includes full local training lifecycle checkpoints with model, optimizer,
minimal scheduler slot, trainer state, RNG state, config/tokenizer snapshots,
adapter/generated metadata, artifact lookup, and CLI resume commands for
TinyTrainer, SFT, and CPT runs. It now also includes a local experiment
registry and matrix/list runner for sequential CPU-safe sweeps with summary
JSON/CSV outputs. It now includes a local benchmark/evaluation suite with
loss, generated-code correctness, router, parameter-efficiency, and composite
CPU-smoke evaluators plus benchmark records and metrics JSON/CSV outputs. It
now includes a local result-analysis layer that loads experiment summaries,
benchmark metrics, and run-result JSON files into normalized rows, simple
comparisons, rankings, deltas, and Markdown reports. It now also includes a
local dataset registry/versioning layer with file fingerprints, manifests,
stats, deterministic splits, split materialization, and dataset-reference
metadata for future reproducible experiment and benchmark runs. It now adds a
local model registry and architecture configuration layer, research run
manifests for future GPU/cloud execution planning, local result import
plumbing, CPU-smoke ablation and baseline comparison helpers, simple
statistical summary tables, and paper-style Markdown report scaffolds. It now
adds a device/precision runtime foundation that keeps CPU as the default while
allowing explicit `cpu`, `auto`, `cuda`, `cuda:N`, and `mps` runtime planning,
dry-runs, metadata, and optional tiny single-device smoke execution when the
local PyTorch install supports it.
It now includes a serious single-GPU research beta: GPU-aware training configs,
AMP/GradScaler hooks, gradient accumulation, activation-checkpointing metadata,
GPU data-loader plumbing, checkpoint/resume, memory estimates, large job
profiles, torchrun dry-run planning, and MoP/Fast-Parameter execution metadata.
This is not yet a fully production distributed LLM training framework.

## What Was Built

The implemented `mopforge.kts` package provides a small file-based database for
structured training lessons:

- `KnowledgeLesson`: a strict standard-library dataclass schema with validation.
- `LessonStore`: a JSONL-backed store that appends, loads, validates, filters,
  counts, looks up by ID, rejects duplicate IDs by default, and samples lessons.
- `filter_lessons`: utilities for domain, skill, subskill, module, difficulty,
  verification, and metadata filtering.
- `LessonDataset`: a framework-agnostic dataset that returns plain dictionaries.
- `TorchLessonDataset`: an optional PyTorch dataset wrapper when PyTorch is
  installed; otherwise it is `None` and the package still imports normally.
- `verify_python_solution`: a minimal local Python subprocess verifier placeholder.
- `VerificationResult`: a typed verifier result with exit code, stderr/stdout,
  timeout, coarse failure type, and elapsed duration.
- `mopforge.builders.coding_bugfix`: deterministic generation of verified
  Python bug-fix lessons.
- `ByteTokenizer`: deterministic byte-level tokenizer with fixed special tokens.
- `format_lesson_for_causal_lm`: prompt/target/full-text lesson formatting.
- `LessonCausalLMDataset`: model-ready causal-LM examples with prompt masking.
- `CausalLMCollator`: optional PyTorch padding collator.
- `TinyCausalTransformer`: optional tiny PyTorch causal transformer baseline for
  smoke testing the data pipeline.
- `TinyMoPCausalTransformer`: optional tiny oracle-routed MoP causal LM that
  activates module-specific parameters from lesson `target_modules`.
- `normalize_target_modules` and `module_mask_from_targets`: stable oracle
  routing helpers for module-targeted lessons.
- `RouterDataset` and `RouterCollator`: deterministic task-text examples and
  padded CPU-safe batches for learned router training.
- `TinyModuleRouter`: optional tiny PyTorch multi-label router that predicts
  active modules from task text.
- `predict_modules` and `route_batch_with_router`: helpers for converting
  router logits into normalized module names and feeding them into TinyMoP.
- `TinyExperimentConfig`, `run_tiny_comparison`, and `write_results`: a tiny
  CPU-safe harness for dense vs oracle-MoP vs learned-router-MoP comparisons.
- `generate_greedy`, `extract_python_code`, and
  `evaluate_generated_code_for_lesson`: a tiny generated-code evaluation path
  from model output to verifier pass/fail.
- `RepairFailureRecord`, `build_repair_lesson_from_failure`, and
  `build_repair_lessons_from_generation_results`: failure-to-lesson conversion
  for repair curriculum data.
- `LessonIndex` and `IndexedLessonStore`: a serverless SQLite metadata index
  and query layer over canonical JSONL lesson records.
- `CurriculumConfig`, `CurriculumScheduler`, and `CurriculumPlan`: deterministic
  curriculum planning over indexed lessons.
- `TinyTrainingRunConfig`, `TrainingRunRecord`, and `RunRegistry`: CPU-safe
  curriculum-driven training records under `runs/<run_id>/`.
- `run_tiny_training_from_curriculum`: bridge from indexed KTS curriculum plans
  to tiny dense/TinyMoP training metrics.
- `LessonFeedbackRecord` and `LessonFeedbackStore`: per-lesson generated/eval
  outcome records in a local SQLite feedback database.
- `score_lesson` and `rank_lesson_ids_by_feedback`: deterministic heuristics
  for prioritizing lessons with more failures, attempts, and loss.
- `feedback_weighted` curriculum strategy: offline curriculum ordering from
  feedback summaries, with deterministic fallback when no feedback DB exists.
- `FeedbackRetrainingConfig`, `FeedbackRetrainingResult`, and
  `run_feedback_retraining_loop`: a tiny CPU-safe closed loop from feedback DB
  to feedback-weighted retraining, generated-code evaluation, and new feedback.
- `TrainingQueueItem`, `TrainingQueueStore`, and queue builders: local
  SQLite-backed module-specific training queue metadata.
- `consume_queue_once`: a tiny local queue consumer that claims one pending
  item, writes queue metadata, and marks it done.
- `ArtifactRecord`, `ArtifactManager`, and `CheckpointManager`: local JSONL
  artifact manifest management and tiny optional PyTorch state-dict
  checkpoint save/load helpers.
- `TrainerConfig`, `TrainerState`, `TrainerResult`, and `TinyTrainer`: a
  reusable CPU-first trainer skeleton for dense, oracle TinyMoP, and
  learned-router TinyMoP smoke training with checkpoints and artifacts.
- `TrainableParameterPolicy`, `ParameterGroupSummary`, and policy helpers:
  name-based parameter grouping, freeze/unfreeze policies, trainable-only
  AdamW optimizer construction, queue-item-to-policy mapping, and trainer
  parameter-count metadata.
- `FastAdapterConfig`, `FastAdapter`, and `FastAdapterBank`: optional tiny
  named bottleneck adapters that can be attached to tiny models, selected from
  target modules, trained with `fast_adapters_only`, and reported in trainer
  metadata.
- `TrainingModeSpec`, `FinetuneConfig`, `FinetuneResult`, and `run_finetune`:
  a CPU-smoke FT/SFT mode API for full SFT, module SFT, adapter SFT, router SFT
  metadata/pathing, repair SFT, and continued-pretraining smoke runs.
- `TextCorpusRecord`, `TextCorpusStore`, `CorpusCausalLMDataset`, and
  `run_continued_pretraining`: continued-pretraining corpus records, JSONL
  storage, full-sequence causal-LM chunks, and a tiny CPU CPT runner with
  artifacts/checkpoints.
- `TokenizerSpec`, `TokenizerProtocol`, `build_tokenizer`, and
  `tokenizer_spec_from_config`: a serializable tokenizer configuration and
  registry/factory layer. `ByteTokenizer` remains the default and can be built
  from a spec.
- `HFTokenizerWrapper`: optional local Hugging Face/tokenizers compatibility
  without mandatory dependencies, downloads, or internet access in tests.
- Generic tokenizer compatibility for SFT/CPT/router datasets and collators,
  including graceful handling when a tokenizer lacks BOS/EOS IDs.
- `GeneratedParameterConfig`, `ConditionEmbedding`, and `GeneratedAdapter`: a
  tiny generated-parameter/hypernetwork path that maps named conditions to
  per-forward low-rank adapter tensors or scale/shift vectors.
- `condition_names_from_target_modules`: a simple mapping from lesson target
  modules to generated-parameter conditions for trainer routing.
- `generated_params_only` trainable policy mode: freezes base tiny model
  weights and trains only condition embeddings plus hypernetwork parameters.
- `MoPForgeConfig`, config IO helpers, default templates, validation, and
  dry-run summaries: JSON always works, YAML works when optional PyYAML is
  installed, and runtime mapping rejects unknown payload fields.
- `mopforge` CLI: `version`, `modes list`, config write/validate/dry-run, and
  local CPU SFT/CPT/TinyTrainer run and resume commands from config files or
  full checkpoint paths/run IDs.
- `TrainingCheckpointRecord`, `capture_rng_state`,
  `save_full_training_checkpoint`, and `load_full_training_checkpoint`: local
  full-checkpoint lifecycle primitives for model/optimizer/scheduler-slot
  state, trainer state, RNG state, config, tokenizer, policy, adapter, and
  generated-parameter metadata.
- `CheckpointManager.save_full_training_checkpoint` and
  `latest_full_checkpoint`: artifact-managed full checkpoint registration under
  `artifacts/checkpoints/` with manifest metadata for optimizer/scheduler/RNG
  presence and global step.
- TinyTrainer/SFT/CPT full resume: new runs can load a full checkpoint, restore
  model and optimizer state, restore Python/NumPy/PyTorch CPU RNG state when
  available, continue `global_step`, and report resume metadata and full
  checkpoint artifact IDs.
- `ExperimentConfig`, `expand_experiment_matrix`, `ExperimentRegistry`, and
  `run_experiment`: local CPU experiment orchestration for matrix/list configs,
  sequential child runs, experiment records, per-run records, and summary
  JSON/CSV files.
- `mopforge experiment`: CLI commands for experiment dry-runs, sequential runs,
  listing local experiment records, and showing one experiment.
- `BenchmarkConfig`, `BenchmarkRegistry`, evaluator helpers, and
  `run_benchmark`: local CPU benchmark plumbing for loss, generated-code
  correctness, router predictions, parameter efficiency, and composite smoke
  evaluations.
- `mopforge benchmark`: CLI commands for benchmark dry-runs, runs, local record
  listing, and showing one benchmark record.
- `mopforge.runtime`: runtime config, device detection/resolution, precision
  policy, autocast helpers, deterministic best-effort hooks, nested batch/model
  movement helpers, and JSON-safe runtime metadata.
- `mopforge runtime`: CLI commands for local device inventory and runtime
  dry-runs that work on CPU-only machines and can plan optional CUDA/MPS smoke
  paths when available.
- `mopforge.gpu`: single-device GPU research beta with CPU fallback,
  `GPUTrainingConfig`, `GPUTrainer`, AMP scaler wrapper, gradient accumulation,
  GPU data loaders, checkpoint/resume, memory estimator, torchrun dry-run
  command generation, run registry, and MoP routing/Fast-Parameter metadata.
- `mopforge gpu`: CLI commands for validating, estimating, training, resuming,
  benchmarking, listing/showing GPU runs, and printing torchrun dry-run plans.
- Example scripts for creating and filtering demo lessons.
- Pytest coverage for the schema, store, filters, sampling, dataset wrappers,
  verifier, coding bug-fix builder, tokenizer, formatter, causal-LM data path,
  optional tiny dense model, routing helpers, optional tiny MoP model, and tiny
  learned router, tiny comparison harness, and generated-code evaluation.
  Repair-loop tests cover failure records, repair lesson construction, and JSONL
  storage. SQLite index tests cover schema creation, rebuild, queries, counts,
  grouped stats, and indexed writes. Curriculum tests cover deterministic
  strategies, filters, batching, lesson loading, and JSON plan output.
  Run-registry tests cover run records, file-backed saves/loads, and tiny
  curriculum-driven dense/oracle-MoP training. Feedback tests cover records,
  SQLite summaries, generated-eval imports, scoring, feedback-weighted
  scheduling, fallback behavior, and JSON export. Feedback-loop tests cover
  config defaults, loop-result JSON, tiny retraining, feedback appends,
  artifact output, and CPU-only assumptions. Queue tests cover item schema,
  SQLite storage, deterministic claiming, status transitions, module counts,
  curriculum/indexed-store builders, consumer behavior, and JSON export.
  Artifact tests cover record validation, manifest registration, duplicate
  rejection, filters, copying, manifest export, tiny checkpoint save/load,
  latest-checkpoint selection, and CPU-only assumptions. Trainer tests cover
  config defaults, state/result serialization, setup, dense/oracle/learned
  router smoke training, checkpoint save/load, resume metadata, artifact
  manifest entries, parameter policy metadata, target-module trainer smoke
  training, and CPU-only assumptions. Parameter policy tests cover validation,
  group inference, train/freeze modes, TinyMoP module-bank selection,
  trainable-only optimizer construction, queue policy mapping, and CPU-only
  assumptions. Fast-adapter tests cover config validation, adapter/bank shape,
  named and multiple adapter selection, unknown-name behavior, target-module
  mapping, TinyMoP forward paths, adapter-only policy freezing, optimizer
  filtering, trainer metadata, and CPU-only assumptions. FT/SFT mode tests
  cover mode specs, validation, mode-to-trainer mappings, one-step full/module/
  adapter SFT runs, result JSON output, mode metadata, and CPU-only assumptions.
  Continued-pretraining tests cover corpus record validation, JSONL storage,
  duplicate rejection, lesson/demo corpus builders, deterministic chunking,
  unmasked full-sequence labels, result/metrics artifacts, checkpoint
  registration, one-step CPT training, and CPU-only assumptions. Tokenizer
  abstraction tests cover spec validation/JSON round-trips, byte-tokenizer
  factory compatibility, old import paths, Unicode round-trips, generic
  tokenizer datasets/collators, SFT/CPT tokenizer spec artifacts, optional HF
  dependency errors, and CPU-only assumptions. Generated-parameter tests cover
  config validation, condition normalization/mapping, condition embeddings,
  generated adapter shape/determinism, TinyMoP forward integration with and
  without fast adapters, parameter grouping/freezing, optimizer construction,
  one-step TinyTrainer/SFT/CPT runs, checkpoint metadata, and CPU-only
  assumptions. Config/CLI tests cover envelope round-trips, JSON/YAML IO,
  default template validation, runtime mapping, validation errors, dry-runs,
  CLI version/modes/config commands, one-step CLI SFT and CPT runs, and
  CPU-only assumptions. Lifecycle/resume tests cover RNG capture/restore,
  full checkpoint save/load payloads, manifest registration, latest full
  checkpoint lookup, TinyTrainer/SFT/CPT resume, optimizer/RNG restore
  metadata, CLI train/SFT/pretrain resume commands, config checkpoint
  validation, and CPU-only assumptions. Experiment tests cover config
  validation, matrix/list expansion, deterministic ordering, max-run limiting,
  invalid dotted paths, registry persistence, summary JSON/CSV output,
  successful and failing child runs, config envelope mapping, dry-runs, CLI
  experiment commands, default experiment configs, and CPU-only assumptions.
  Benchmark tests cover config validation, registry persistence, metric helper
  behavior, loss/code/router/parameter/composite evaluators, runner artifacts,
  failed-evaluator capture, config envelope mapping, dry-runs, CLI benchmark
  commands, default benchmark configs, and CPU-only assumptions. Runtime tests
  cover config validation and round-trips, device detection/resolution,
  precision fallback policy, runtime context metadata, recursive batch movement,
  model movement, TinyTrainer/SFT/CPT/benchmark runtime smoke paths, runtime
  config defaults, CLI runtime commands, and optional CUDA skips.

## Why Structured Lessons

A future Mixture-of-Parameters model may have separate parameter groups such as
`core`, `coding`, `debugging`, `math`, `planning`, `router`, and
`fast_adapter`. Random raw text does not say which module should learn which
behavior, how hard the example is, or whether the answer was verified.

Knowledge Training Store lessons encode those signals directly: concept, domain,
skill, target modules, difficulty, expected output, verification metadata, and
common failures. That makes the data usable by PyTorch, NumPy, or custom
training loops without committing this project to a specific model architecture.

## Installation

```bash
pip install -e .
```

For local test development:

```bash
pip install -e .[dev]
```

## Basic Usage

```python
from mopforge.kts import KnowledgeLesson, LessonDataset, LessonStore

store = LessonStore("data/lessons.jsonl")

lesson = KnowledgeLesson(
    id="debug-missing-return-001",
    domain="coding",
    skill="debugging",
    subskill="missing-return",
    difficulty=2,
    target_modules=["coding", "debugging"],
    input="def add(a, b):\n    a + b",
    expected_output="def add(a, b):\n    return a + b",
    verification={"type": "python_tests", "status": "verified"},
    metadata={"language": "python"},
)

store.add(lesson)
lessons = store.filter(domain="coding", target_modules=["debugging"])
dataset = LessonDataset(lessons)
sample = dataset[0]
```

## Goal 2: Lesson Builder + Verified Coding Dataset

The Goal 2 builder turns passive storage into a small lesson generation and
verification pipeline for Python debugging examples. It generates deterministic
`KnowledgeLesson` records for these bug categories:

- missing return
- off-by-one loop/index bug
- wrong comparison operator
- wrong accumulator initialization
- incorrect base case in recursion

Every generated lesson has stable IDs, `domain="coding"`, `skill="debugging"`,
module targets including `coding` and `debugging`, buggy code as `input`, fixed
code as `expected_output`, verification metadata, and metadata for language,
function name, bug type, variant index, test names, and test code.

Generate the verified demo dataset:

```bash
python examples/generate_coding_bugfix_lessons.py
```

The script creates `data/coding_bugfix_lessons.jsonl`, writes 50 verified
lessons, and prints counts by bug category and verification status.

Programmatic generation:

```python
from mopforge.builders import generate_coding_bugfix_lessons

lessons = generate_coding_bugfix_lessons(count_per_category=10)
```

Verification runs the fixed solution plus generated tests in a temporary local
Python subprocess. `VerificationResult` captures stdout, stderr, exit code,
timeout state, duration, and coarse error type such as `syntax_error`,
`runtime_error`, `test_failure`, or `timeout`. These fields are copied into each
lesson's verification metadata.

These lessons are intended to train future coding/debugging modules by making
the target behavior explicit: what failed, which modules should learn from it,
which tests verified it, and which common failure pattern it prevents.

## Goal 3: Model-Ready Data Pipeline + Tiny Dense Baseline

Goal 3 adds the first model-facing layer without introducing the MoP model yet.
Structured `KnowledgeLesson` records can now be converted into supervised
causal-LM examples.

The data path includes:

- `ByteTokenizer`: a no-training UTF-8 byte tokenizer with `<pad>`, `<bos>`, and
  `<eos>` IDs. It does not require Hugging Face or `tokenizers`.
- `format_lesson_for_causal_lm`: deterministic prompt/target formatting where
  the prompt contains task context and the target is only the expected output.
- `LessonCausalLMDataset`: tokenizes lessons into `input_ids`, `labels`, and
  `attention_mask`. Prompt labels are masked with `-100` by default so loss is
  applied only to the target side.
- `CausalLMCollator`: optional PyTorch collator that pads `input_ids` with the
  tokenizer pad ID, labels with `-100`, and attention masks with `0`.
- `TinyCausalTransformer`: optional small PyTorch causal transformer with token
  embeddings, positional embeddings, causal self-attention, LM head, and
  cross-entropy loss for smoke tests.

Tokenize one generated lesson:

```bash
python examples/tokenize_lessons.py
```

Run a tiny CPU smoke-training loop when PyTorch is installed:

```bash
python examples/train_tiny_dense_baseline.py
```

The tiny baseline is only a pipeline test. It is not a production model and the
printed losses should not be interpreted as meaningful model performance.

## Goal 4: Tiny Oracle-Routed MoP Model

Goal 4 adds the first actual Mixture-of-Parameters prototype. It is deliberately
small and CPU-only by default for architecture smoke testing on modest hardware.

The model path includes:

- `TinyMoPCausalTransformer`: token embeddings, positional embeddings, shared
  causal transformer blocks, a bank of small module-specific MLP blocks, and an
  LM head.
- Oracle routing from `KnowledgeLesson.target_modules`, passed through the
  causal-LM collator as `batch["target_modules"]`.
- `normalize_target_modules`: removes duplicates, ignores unknown modules by
  default, and always includes `core` when available.
- `module_mask_from_targets`: creates a stable 0/1 mask aligned to known module
  names.

The minimal routing formula is:

```text
h = shared_blocks(x)
module_delta = mean(module_block(h) for active module)
h = h + module_delta
logits = lm_head(norm(h))
```

Run the TinyMoP CPU smoke test:

```bash
python examples/train_tiny_mop_baseline.py
```

The example uses tiny settings: `d_model=64`, `n_layers=2`, `n_heads=2`,
`max_seq_len=512`, `batch_size=2`, and `training_steps=3`. It does not require
CUDA and does not attempt meaningful model training.

## Goal 5: Learned Router MVP

Goal 5 adds the first learned router path while preserving oracle routing. The
router learns a tiny multi-label mapping:

```text
structured lesson/task text -> module mask -> active MoP modules
```

The router path includes:

- `RouterDataset`: converts `KnowledgeLesson` records into deterministic route
  prompts using domain, skill, subskill, difficulty, concept, and input text.
  It intentionally omits `expected_output` so the router learns from the task,
  not the answer.
- `RouterCollator`: optional PyTorch collator that pads route `input_ids` and
  stacks multi-label `module_mask` tensors.
- `TinyModuleRouter`: token embedding, masked mean pooling, small MLP, and
  `BCEWithLogitsLoss` for multi-label module prediction.
- `predict_modules`: converts router logits into normalized module names and
  keeps `core` active when requested.
- `route_batch_with_router`: small integration helper for using learned router
  predictions as TinyMoP `active_modules`.

Train the tiny router CPU smoke test:

```bash
python examples/train_tiny_router.py
```

Optionally smoke-test TinyMoP with learned-router predictions:

```bash
python examples/train_tiny_mop_with_learned_router.py
```

Both examples use tiny CPU settings: `batch_size=2`, `training_steps=3` or less,
`max_seq_len=512`, `d_model=64`, and no multiprocessing. Printed losses are only
architecture smoke-test signals, not router accuracy claims.

## Goal 6: Tiny Experiment Harness

Goal 6 adds a reproducible CPU-safe comparison path for the three current model
routes:

- `tiny_dense` with no routing.
- `tiny_mop` with oracle routing from lesson `target_modules`.
- `tiny_mop` with modules predicted by the learned router.

The harness uses the same verified coding/debugging lessons, the same
`ByteTokenizer`, and the same tiny CPU config for each path. It reports plain
dictionaries with fields such as `model`, `routing`, `train_loss_last`,
`eval_loss_mean`, `finite`, `train_examples`, and `eval_examples`. Learned
router runs also include simple routing metrics, such as exact-match count and
average predicted/target module counts.

Run the comparison:

```bash
python examples/run_tiny_comparison.py
```

The script prints a small table and writes:

```text
outputs/tiny_comparison_results.json
outputs/tiny_comparison_results.csv
```

This is only measurement plumbing. It does not prove that MoP is better than a
dense model, and the losses are not meaningful model-quality claims yet.

## Goal 7: Generated Code Evaluation

Goal 7 adds the first correctness-evaluation loop. Loss-only comparison is not
enough for coding tasks, so the framework can now ask a tiny model to produce
text, extract candidate code, and run the lesson's tests through the existing
local verifier.

The generated-code path includes:

- `format_lesson_prompt`: stable prompt-only formatting that excludes
  `expected_output`.
- `generate_greedy`: CPU-safe greedy autoregressive generation for
  `TinyCausalTransformer` and `TinyMoPCausalTransformer`.
- `extract_python_code`: simple deterministic extraction for fenced Python
  blocks, generic fenced blocks, and raw code.
- `evaluate_generated_code_for_lesson`: prompt, generate, extract, verify, and
  return pass/fail metadata.
- Optional generation metrics in the tiny comparison harness:
  `gen_eval_examples`, `gen_pass_count`, `gen_pass_rate`, and
  `gen_failures_by_type`.

Run the generated-code smoke evaluation:

```bash
python examples/evaluate_tiny_generated_code.py
```

The script prints per-lesson pass/fail summaries and writes:

```text
outputs/tiny_generated_code_eval.json
```

The current tiny models are not expected to pass. This is evaluation plumbing
for future repair loops, verifier-guided training, and real dense-vs-MoP
correctness comparisons.

## Goal 8: Repair Loop MVP

Goal 8 turns generated-code failures into structured repair lessons. This is the
first active Knowledge Training Store loop:

```text
failed generated output -> failure record -> repair lesson -> KTS
```

The repair path includes:

- `RepairFailureRecord`: captures source lesson ID, original input, expected
  output, generated text, extracted candidate code, failure type, verifier
  details, target modules, and metadata.
- `build_repair_lesson_from_failure`: creates a validated `KnowledgeLesson`
  with `domain="coding"` and `skill="repair"`. The lesson input includes the
  original task, failed candidate, failure type, verifier output when present,
  and an instruction to repair the code. The expected output remains the known
  verified target from the source lesson.
- `failure_record_from_generation_result`: converts failed generated-code eval
  results into repair failure records.
- `write_repair_lessons`: stores repair lessons in JSONL through `LessonStore`.

Build repair lessons from tiny generated-code eval output:

```bash
python examples/build_repair_lessons_from_tiny_eval.py
```

The script writes:

```text
data/repair_lessons.jsonl
```

Repair lessons do not claim the failed candidate is correct. Their verification
status is `verified_target`, meaning the target answer is known and verified
while the generated candidate failed.

## Goal 9: KTS v2 SQLite Metadata Index

Goal 9 upgrades the Knowledge Training Store into a custom file-backed training
database:

```text
canonical JSONL lesson records
+ SQLite metadata index
+ query/count/rebuild APIs
```

JSONL remains the source of truth. SQLite stores searchable metadata and record
pointers for fast local queries, counts, and future curriculum scheduling.

The index path includes:

- `LessonIndex`: creates the SQLite schema, indexes individual lessons, rebuilds
  from a JSONL store, queries lesson IDs or metadata rows, counts matches, and
  groups counts by domain, skill, subskill, verification status, or target
  module.
- `IndexedLessonStore`: additive wrapper combining `LessonStore` and
  `LessonIndex`. Writes append to JSONL and update SQLite metadata.
- Metadata query support for domain, skill, subskill, difficulty range,
  verification type/status, target modules, and key/value metadata.

Build the demo index:

```bash
python examples/index_kts_lessons.py
```

The script writes:

```text
data/kts_index.sqlite
```

and prints total indexed lessons, grouped counts, debugging-module count, and a
sample query for `verified_target` repair lessons.

## Goal 10: Curriculum Scheduler MVP

Goal 10 makes the indexed KTS act like a deterministic training teacher:

```text
indexed KTS -> query/filter/group/order -> lesson plan -> batches
```

The scheduler uses SQLite metadata from `LessonIndex` and supports these MVP
strategies:

- `sequential`: sorted deterministic lesson IDs.
- `shuffled`: deterministic random order with a seed.
- `balanced`: round-robin balancing by skill.
- `module_targeted`: filters by target modules such as `debugging`.
- `repair_boosted`: places repair-like lessons first, where repair-like means
  `skill == "repair"` or `verification_status == "verified_target"`.

Plans include lesson IDs, counts by skill/domain/verification status/target
module, total count, and metadata. They can be saved as JSON and loaded back
into `KnowledgeLesson` objects through `CurriculumScheduler.load_lessons`, ready
for `LessonCausalLMDataset`.

Build demo curriculum plans:

```bash
python examples/schedule_curriculum.py
```

The script writes:

```text
outputs/curriculum_plan_balanced.json
outputs/curriculum_plan_repair_boosted.json
```

This is deterministic scheduling plumbing, not adaptive learning yet.

## Goal 11: Curriculum-Driven Training Runner

Goal 11 adds the first run loop over the indexed KTS and curriculum scheduler:

```text
KTS/index -> curriculum plan -> loaded lessons -> tiny training -> run record
```

The runner path includes:

- `TinyTrainingRunConfig`: tiny CPU-safe run settings such as model type,
  curriculum strategy, batch size, train steps, eval batches, and optional
  generated-code eval.
- `TrainingRunRecord`: persisted run metadata with config, model type,
  curriculum strategy, metrics, timestamps, and artifact paths.
- `RunRegistry`: file-backed registry rooted at `runs/`, storing:

```text
runs/
  <run_id>/
    run.json
    metrics.json
    curriculum_plan.json
```

- `run_tiny_training_from_curriculum`: builds a curriculum plan, loads lessons,
  runs one tiny dense/TinyMoP training path, computes train/eval metrics, and
  registers the run.

Supported model types:

- `dense`
- `mop_oracle`
- `mop_learned_router`

Run the CPU smoke training registry example:

```bash
python examples/run_curriculum_training.py
```

The example runs:

- dense + balanced curriculum
- oracle TinyMoP + repair-boosted curriculum

Losses and pass rates are not meaningful quality claims. This is the bridge to
future adaptive scheduling from live model metrics.

## Goal 12: Feedback-Aware Curriculum MVP

Goal 12 adds the first offline feedback layer:

```text
generated eval failures -> feedback DB -> lesson scores -> prioritized curriculum
```

The feedback path includes:

- `LessonFeedbackRecord`: one per-lesson event with optional run/model/strategy,
  pass/fail, failure type, loss, generated flag, timestamp, and metadata.
- `LessonFeedbackStore`: SQLite store at paths such as
  `data/lesson_feedback.sqlite`.
- `lesson_feedback` table: raw event rows.
- `lesson_feedback_summary` table: attempts, passes, failures, average loss,
  last failure type, and last seen timestamp per lesson.
- `feedback_records_from_generation_eval`: imports Goal 7 generated-code eval
  reports, including the grouped JSON written by
  `examples/evaluate_tiny_generated_code.py`.
- `feedback_weighted` scheduler strategy: sorts eligible lessons by feedback
  score descending, then by lesson ID for deterministic ties.

Scoring is intentionally simple and inspectable: failed lessons score higher
than passed lessons, attempts and average loss add smaller weight, and unseen
lessons get a small default score so they remain eligible.

Run the feedback-aware curriculum example:

```bash
python examples/feedback_aware_curriculum.py
```

The script writes:

```text
data/lesson_feedback.sqlite
outputs/lesson_feedback_export.json
outputs/curriculum_plan_feedback_weighted.json
```

These priorities come from tiny CPU smoke outputs and are not model-quality
claims. This is the bridge to future adaptive scheduling, not a live adaptive
trainer.

## Goal 13: Feedback-Weighted Retraining Loop MVP

Goal 13 adds the first closed CPU-smoke feedback loop:

```text
feedback DB -> feedback-weighted curriculum -> tiny retraining
    -> generated-code eval -> new feedback records -> loop report
```

The loop path includes:

- `FeedbackRetrainingConfig`: tiny settings for model type, lesson/index paths,
  feedback DB path, run registry root, training steps, generated eval examples,
  and model size.
- `FeedbackRetrainingResult`: loop ID, training run ID, feedback records added,
  generated eval pass/fail counts, failures by type, artifact paths, and metrics.
- `run_feedback_retraining_loop`: builds a feedback-weighted curriculum, runs
  the same tiny curriculum training path used by Goal 11, evaluates generated
  code on a deterministic tiny subset, imports those outcomes into the feedback
  DB, and writes loop artifacts.
- `summarize_feedback_delta`: small before/after feedback-count and
  failure-count comparison helper.

Loop artifacts are stored under:

```text
runs/<loop_id>/
  loop_result.json
  generation_eval_after_retraining.json
  feedback_export_after_retraining.json
```

The training run itself is also recorded through `RunRegistry` under
`runs/<train_run_id>/`.

Run the feedback-retraining loop example:

```bash
python examples/run_feedback_retraining_loop.py
```

This is not evidence of self-improvement. It only proves the plumbing can take
stored feedback, prioritize lessons, run a tiny training pass, evaluate
generated output, and append new feedback records.

## Goal 14: Module-Specific Training Queues MVP

Goal 14 adds local, inspectable module-specific training queues:

```text
indexed KTS + curriculum/feedback -> per-module queue items
  -> queue store -> tiny queue consumer -> queue metadata
```

The queue path includes:

- `TrainingQueueItem`: one scheduled unit of module-targeted work with module,
  lesson ID, priority, status, source, run ID, attempts, timestamps, and
  metadata.
- `TrainingQueueStore`: SQLite store for `training_queue` rows with indexes on
  module, status, priority, and lesson ID.
- `build_queue_items_from_curriculum`: turns a curriculum plan and loaded
  lessons into one queue item per target module, defaulting to `core` if a
  lesson has no target modules.
- `build_module_queue_from_indexed_store`: builds a curriculum plan from an
  `IndexedLessonStore`, optionally scores lesson priority from feedback, and
  returns module-specific queue items.
- `consume_queue_once`: non-production local smoke consumer. Dry runs preview
  without claiming. Normal runs claim the highest-priority pending item, write
  a small `queue_item.json`, and mark the item done.

Queue data is stored locally:

```text
data/training_queue.sqlite
outputs/training_queue_export.json
runs/<queue_run_id>/queue_item.json
```

Run the module queue example:

```bash
python examples/build_module_training_queue.py
```

Queue items are scheduling metadata only. The consumer does not run long
training or make quality claims; it proves deterministic queue mechanics for
future module-specific training work.

## Goal 15: Checkpoint + Artifact Manager MVP

Goal 15 adds a local artifact lifecycle layer:

```text
file -> artifact record -> manifest
model state_dict -> checkpoint artifact -> load/list/latest
```

The artifact path includes:

- `ArtifactRecord`: metadata for one local file, including kind, path, run ID,
  queue item ID, model type, module, step, timestamp, and custom metadata.
- `ArtifactManager`: JSONL manifest at `artifacts/manifest.jsonl`, with helper
  directories for checkpoints, metrics, evaluations, configs, and other files.
- `copy_artifact`: copies a local file under the artifact root and registers it
  in the manifest.
- `CheckpointManager`: optional PyTorch helper for saving/loading tiny
  `state_dict` checkpoints, listing checkpoints, and selecting the latest by
  step and timestamp.

The default layout is:

```text
artifacts/
  manifest.jsonl
  checkpoints/
  metrics/
  evaluations/
  configs/
  other/
```

Run the artifact/checkpoint example:

```bash
python examples/manage_artifacts_and_checkpoints.py
```

The example registers a small JSON artifact, saves a tiny local checkpoint,
loads it into a fresh tiny model, lists the latest checkpoint, and exports:

```text
outputs/artifact_manifest_export.json
```

This is local reproducibility plumbing only, not a model hub or production
checkpoint lifecycle.

## Goal 16: Production-Style Trainer Skeleton, CPU-First

Goal 16 adds a reusable trainer skeleton:

```text
TrainerConfig -> TinyTrainer -> TrainerResult
  + curriculum
  + CPU-safe train/eval
  + checkpoints
  + artifacts
  + run registry
```

The trainer path includes:

- `TrainerConfig`: one config object for model type, routing mode, lesson/index
  paths, feedback/queue paths, curriculum strategy, batch/train/eval settings,
  CPU device settings, run/artifact roots, checkpoint settings, and optional
  generation eval.
- `TrainerState`: resumable progress metadata including global step, best eval
  loss, latest losses, checkpoint artifact IDs, and metric history.
- `TrainerResult`: final run ID, model/routing info, final state, metrics,
  artifacts, and finite status.
- `TinyTrainer`: setup, train, evaluate, save/load checkpoint, artifact
  registration, and run-record writing.

Supported tiny model paths:

- `dense`: `TinyCausalTransformer`
- `mop_oracle`: `TinyMoPCausalTransformer` with lesson target modules
- `mop_learned_router`: TinyMoP with a tiny learned router smoke path

Run output is written under:

```text
runs/<run_id>/
  run.json
  metrics.json
  trainer_result.json
  trainer_state.json
  curriculum_plan.json
```

Checkpoints are registered through `ArtifactManager` and saved under
`artifacts/checkpoints/`. The original tiny state-dict checkpoint path remains,
and Goal 24 adds full lifecycle checkpoints with optimizer state, trainer
state, RNG state, config/tokenizer snapshots, and a nullable scheduler slot.

Run the trainer example:

```bash
python examples/run_tiny_trainer.py
```

This is trainer architecture plumbing only. Metrics and checkpoint files are
CPU-smoke artifacts, not model-quality claims.

## Goal 17: Module-Specific Training Policies + Freezing MVP

Goal 17 adds trainable-parameter control for tiny CPU trainer runs:

```text
model -> inferred parameter groups -> freeze/unfreeze policy
  -> optimizer over trainable params -> trainer metrics/artifacts
```

The policy path includes:

- `ParameterGroupSummary`: total/trainable/frozen counts for one inferred
  parameter group.
- `TrainableParameterPolicy`: supported modes are `all`, `core_only`,
  `modules_only`, `target_modules_only`, `router_only`, `head_only`,
  `fast_adapters_only`, and `frozen`.
- `infer_parameter_group`: best-effort name-based grouping for tiny model
  parameters such as `embeddings`, `shared_core`, `module:<name>`, `router`,
  `lm_head`, `norm`, and `other`.
- `apply_trainable_policy`: applies `requires_grad` flags and returns group
  summaries.
- `build_optimizer_for_trainable_parameters`: constructs AdamW only over
  parameters that remain trainable.
- `policy_from_queue_item`: maps a local queue item to a
  `target_modules_only` policy for its module.
- `TinyTrainer` integration: policy fields in `TrainerConfig`, policy
  application after model creation, trainable-only optimizer setup, and
  parameter counts/summaries in trainer state, metrics, checkpoints, and run
  metadata.

For the current `TinyMoPCausalTransformer`, module groups are detected from
names like `module_bank.blocks.coding...`, so local CPU smoke runs can freeze
the shared core and train only the `coding` or `debugging` module block.

Run the module-specific policy example:

```bash
python examples/train_module_specific_policy.py
```

This example runs one tiny oracle-MoP trainer pass for `coding` and one for
`debugging`, each with `trainable_policy_mode="target_modules_only"`, and
prints run IDs, parameter counts, group summaries, and checkpoint artifact IDs.
It is policy plumbing only. The metrics are not model-quality claims.

## Goal 18: Fast Adapter MVP

Goal 18 adds the first fast-parameter mechanism, without generated parameters
or hypernetworks:

```text
base hidden states + selected fast adapter delta -> adapted hidden states
```

The adapter path includes:

- `FastAdapterConfig`: small adapter dimensions, names, dropout, and residual
  scale validation.
- `FastAdapter`: a tiny LayerNorm + down-projection + GELU + dropout +
  up-projection residual adapter.
- `FastAdapterBank`: a `ModuleDict` of named adapters. `active_adapters=None`
  applies no adapter, unknown adapter names are ignored, and multiple active
  adapters are combined by averaging their deltas.
- `normalize_adapter_names` and `adapter_names_from_target_modules`: helpers
  for stable adapter-name cleanup and simple mapping from lesson target modules
  such as `coding`, `debugging`, `repair`, `router`, and `fast_adapter`.
- `TinyMoPCausalTransformer` and `TinyCausalTransformer` opt-in constructor
  flags: `use_fast_adapters`, `fast_adapter_names`, and
  `fast_adapter_bottleneck_dim`.
- `TrainerConfig` adapter fields: `use_fast_adapters`, `fast_adapter_names`,
  `fast_adapter_bottleneck_dim`, `active_adapters`, and
  `adapter_from_target_modules`.
- `fast_adapters_only` trainable policy mode plus `adapter:<name>` parameter
  grouping, so the trainer can freeze the base model and train only adapter
  parameters.
- Existing non-adapter policy modes keep adapter parameters frozen unless
  `train_fast_adapters=True`; `fast_adapters_only` is the explicit adapter-only
  training mode.

By default, trainer adapter routing uses the union/per-example lesson
`target_modules` already present in batches and maps those modules to adapter
names. Static adapter selection is also available by setting
`adapter_from_target_modules=False` and `active_adapters=[...]`.

Run the fast-adapter smoke example:

```bash
python examples/train_fast_adapter.py
```

The example runs one tiny oracle-MoP trainer pass with adapters named
`coding`, `debugging`, and `repair`, trains only adapter parameters, prints
parameter counts and adapter group summaries, and saves a tiny checkpoint.
It is architecture plumbing only. The metrics are not model-quality claims.

## Goal 19: FT/SFT Training Mode API MVP

Goal 19 formalizes tiny fine-tuning and supervised fine-tuning modes:

```text
KnowledgeLesson input -> expected_output
  -> training mode config
  -> lesson filters + parameter policy
  -> TinyTrainer
  -> FT/SFT run metadata
```

The FT/SFT path includes:

- `TrainingModeSpec`: static metadata for each mode, including objective,
  expected policy mode, expected model type, and mode requirements.
- `list_training_modes` and `get_training_mode_spec`: stable mode discovery.
- `FinetuneConfig`: CPU-safe user config for mode, model type, target modules,
  lesson/index paths, curriculum filters, adapter settings, tiny train
  settings, and artifact roots.
- `trainer_config_from_finetune_config`: maps FT/SFT modes into `TrainerConfig`.
- `build_finetune_lesson_filter`: deterministic domain, skill, verification,
  and target-module filter construction.
- `FinetuneResult` and `run_finetune`: run TinyTrainer, wrap trainer output,
  write `finetune_result.json`, and include mode metadata in metrics/artifacts.

Supported modes:

- `sft_full`: full supervised `input -> expected_output`, policy `all`.
- `sft_module`: selected TinyMoP module SFT, policy `target_modules_only`.
- `sft_adapter`: adapter-only SFT, policy `fast_adapters_only`.
- `sft_router`: MVP router SFT path metadata using the existing learned-router
  smoke path.
- `repair_sft`: repair-oriented SFT with `repair_boosted` curriculum and repair
  lesson filtering by default.
- `continued_pretraining_smoke`: explicit tiny causal-LM smoke continuation,
  not real large-scale pretraining.

Goal 20 adds the real corpus API used for continued-pretraining smoke work;
Goal 19 only introduced the mode name and metadata path.

Run the FT/SFT mode example:

```bash
python examples/run_sft_modes.py
```

The example runs tiny CPU smoke passes for full SFT, module SFT, adapter SFT,
and repair SFT when repair lessons are present. It prints mode, run ID, policy
mode, trainable/frozen parameter counts, final train/eval losses, checkpoint
artifact IDs, and the `finetune_result.json` path. This is API and metadata
plumbing only; it is not a model-quality claim.

## Goal 20: Continued Pretraining Corpus API MVP

Goal 20 adds a separate continued-pretraining data path:

```text
raw/semi-structured text records
  -> JSONL corpus store
  -> full-sequence causal-LM chunks
  -> tiny CPT runner
  -> artifacts and run metadata
```

The distinction is intentional:

- SFT uses structured `KnowledgeLesson.input -> expected_output` supervision.
- Continued pretraining uses raw text/code streams with next-token causal-LM
  loss over the full sequence.

The CPT path includes:

- `TextCorpusRecord`: raw/semi-structured text with source, domain, language,
  timestamp, and metadata.
- `TextCorpusStore`: JSONL-backed corpus storage with duplicate-ID rejection,
  deterministic loading, simple filters, and JSON export.
- `build_corpus_from_lessons`: converts existing KTS lessons into raw text/code
  streams for smoke CPT, without using SFT prompt masking.
- `build_demo_code_corpus`: deterministic tiny Python/code explanation corpus.
- `CorpusCausalLMDataset`: tokenizes records into deterministic fixed-length
  causal-LM chunks. Labels match `input_ids` for the full chunk; only collator
  padding uses `-100`.
- `CorpusCausalLMCollator`: pads corpus chunks for optional PyTorch runs.
- `ContinuedPretrainConfig`, `ContinuedPretrainResult`, and
  `run_continued_pretraining`: tiny CPU runner that trains/evaluates for a few
  steps, writes metrics/result/corpus summary JSON, and optionally registers a
  checkpoint artifact.

Run the continued-pretraining smoke example:

```bash
python examples/run_continued_pretraining.py
```

The example rebuilds `data/text_corpus.jsonl`, runs one tiny CPT pass, and
prints run ID, corpus record/chunk counts, final train/eval loss, checkpoint
artifact IDs, and `continued_pretrain_result.json`. This is data/API plumbing
only. It is not real large-scale pretraining and makes no quality claims.

## Goal 21: Tokenizer Abstraction + HF/BPE Compatibility MVP

Goal 21 replaces byte-tokenizer assumptions in the SFT/CPT data path with a
small tokenizer infrastructure layer:

```text
TokenizerSpec
  -> tokenizer registry/factory
  -> ByteTokenizer default or optional HF wrapper
  -> generic SFT/CPT/router datasets
  -> tokenizer_spec.json run artifacts
```

The tokenizer path includes:

- `TokenizerSpec`: JSON-serializable tokenizer metadata with type, name/path,
  vocab size, special token IDs, and arbitrary JSON metadata.
- `TokenizerProtocol`: the expected structural interface for tokenizers:
  `encode`, `decode`, `vocab_size`, `pad_token_id`, and optional BOS/EOS/UNK
  IDs.
- `build_tokenizer`, `tokenizer_spec_from_config`, and
  `register_tokenizer_type`: a tiny registry/factory for `byte`, `hf`, and
  future tokenizer types.
- `ByteTokenizer.to_spec()`: preserves the existing deterministic UTF-8 byte
  tokenizer and old import paths while making it buildable from
  `TokenizerSpec`.
- `HFTokenizerWrapper`: optional `transformers.AutoTokenizer` support with
  `local_files_only=True` by default, plus local `tokenizers` JSON fallback
  when available. Neither dependency is required for normal installation or
  tests.
- Generic dataset/collator support for `LessonCausalLMDataset`,
  `RouterDataset`, and `CorpusCausalLMDataset`. Tokenizers without BOS/EOS IDs
  are handled by omitting those sequence markers; padding uses the tokenizer
  pad ID, falling back to `0` if absent.
- `TrainerConfig`, `FinetuneConfig`, and `ContinuedPretrainConfig` tokenizer
  fields: `tokenizer_type`, `tokenizer_name_or_path`, and
  `tokenizer_spec_path`. A spec path wins over inline fields.
- TinyTrainer and CPT runs write `tokenizer_spec.json` under the run directory,
  register it as a config artifact, and include tokenizer metadata in metrics
  and checkpoint metadata.

Run the tokenizer abstraction demo:

```bash
python examples/tokenizer_abstraction_demo.py
```

The demo builds a byte tokenizer from `TokenizerSpec`, encodes/decodes a Python
snippet, saves and reloads
`outputs/tokenizer_abstraction_demo/tokenizer_spec.json`, creates a small
generic CPT dataset, and skips the optional HF path unless
`MOPFORGE_HF_TOKENIZER_PATH` points to a local tokenizer.

## Goal 22: Generated Parameters / Hypernetwork MVP

Goal 22 adds the first actual generated-parameter mechanism:

```text
condition names
  -> condition embeddings
  -> tiny hypernetwork
  -> per-forward generated adapter tensors
  -> adapted hidden states
```

This is distinct from Goal 18 fast adapters. Fast adapters are normal stored
modules with their own direct parameters; generated adapters create temporary
adapter tensors from a small hypernetwork during each forward pass.

The generated-parameter path includes:

- `GeneratedParameterConfig`: CPU-safe dimensions, condition names, generator
  type, residual scale, and activation validation.
- `ConditionEmbedding`: maps names such as `coding`, `debugging`, `repair`,
  `math`, `planning`, and `default` into conditioning vectors.
- `GeneratedAdapter`: supports `low_rank_adapter` and `scale_shift` modes.
  The default low-rank path generates tiny `down` and `up` adapter weights for
  each active condition and applies:

```text
delta = activation(h @ down_weight) @ up_weight
h_out = h + residual_scale * delta
```

- `normalize_condition_names` and `condition_names_from_target_modules`:
  deterministic condition cleanup and simple target-module routing. Unknown
  names are ignored; `active_conditions=None` applies no generated adapter.
- `TinyCausalTransformer` and `TinyMoPCausalTransformer` opt-in constructor
  fields for generated parameters. Fast adapters and generated adapters can
  coexist in TinyMoP forward passes.
- `TrainerConfig` generated-parameter fields:
  `use_generated_params`, `generated_condition_names`,
  `generated_condition_dim`, `generated_rank`, `generated_type`,
  `active_conditions`, `conditions_from_target_modules`, and
  `train_generated_params`.
- `generated_params_only` trainable policy mode, which freezes the base tiny
  model and trains only `generated_condition_embedding` and `hypernetwork`
  parameters.
- TinyTrainer metrics and checkpoints now include `generated_metadata`, with
  condition names, generator type, rank, active condition mode, and generated
  parameter counts.
- `FinetuneConfig` includes generated-parameter fields and supports
  `sft_generated`, mapped to `generated_params_only`.
- `ContinuedPretrainConfig` can enable generated parameters for tiny CPT smoke
  runs with static active conditions.

Run the generated-parameter smoke example:

```bash
python examples/train_generated_params.py
```

The example runs one tiny oracle-MoP trainer pass with conditions named
`coding`, `debugging`, and `repair`, trains only generated-parameter
hypernetwork/condition parameters, prints parameter counts and generated group
summaries, and saves a tiny checkpoint. It is architecture plumbing only, not a
model-quality claim.

## Goal 23: YAML Config System + CLI Entrypoints MVP

Goal 23 turns the tiny framework into a config-driven local runner:

```text
JSON/YAML config file
  -> MoPForgeConfig envelope
  -> validation / dry-run
  -> typed runtime config
  -> SFT, CPT, or TinyTrainer run
  -> artifacts and run outputs
```

The config/CLI path includes:

- `MoPForgeConfig`: a versioned envelope with `kind`, `payload`, and
  `metadata`. Supported runnable kinds are `trainer`, `sft`, and `pretrain`;
  `experiment` and `queue` are reserved for later.
- `load_config_file` and `save_config_file`: JSON support through the standard
  library and optional YAML support when PyYAML is installed. If YAML is used
  without PyYAML, the error includes an install hint.
- Default CPU-safe templates for `trainer`, `sft_full`, `sft_module`,
  `sft_adapter`, `sft_generated`, `pretrain`, `generated_sft`, and
  `fast_adapter_sft`.
- Runtime mapping helpers:
  `trainer_config_from_envelope`, `finetune_config_from_envelope`, and
  `pretrain_config_from_envelope`. Unknown payload fields are rejected so
  config typos do not silently change runs.
- `validate_config_envelope`: checks known kind, SFT mode, basic positive
  sizes, required target modules, adapter/generated mode flags, tokenizer spec
  paths, string paths, and CPU-device warnings.
- `dry_run_config`: returns the resolved runtime config, warnings/errors,
  expected output roots, and whether the run is locally runnable.
- `mopforge` CLI implemented with standard-library `argparse`.

CLI commands:

```bash
mopforge version
mopforge modes list
mopforge config write-default sft_full outputs/cli_config_demo/sft_full_cpu.json
mopforge config validate outputs/cli_config_demo/sft_full_cpu.json
mopforge config dry-run outputs/cli_config_demo/sft_full_cpu.json
mopforge sft run outputs/cli_config_demo/sft_full_cpu.json
mopforge pretrain run configs/examples/cpt_cpu.json
mopforge train run configs/examples/tiny_trainer_mop_cpu.json
```

Tracked example configs live under:

```text
configs/examples/
  sft_full_cpu.json
  sft_adapter_cpu.json
  sft_generated_cpu.json
  cpt_cpu.json
  tiny_trainer_mop_cpu.json
```

Run the config/CLI demo:

```bash
python examples/run_cli_configs.py
```

The demo writes default configs into `outputs/cli_config_demo/`, validates and
dry-runs them programmatically, then runs one tiny SFT config. It is local
CPU-smoke workflow plumbing only.

## Goal 24: Full Checkpoint Resume + Training Lifecycle MVP

Goal 24 makes local CPU smoke runs resumable through a full checkpoint payload:

```text
model state
  + optimizer state
  + scheduler state slot
  + trainer/global-step state
  + Python/NumPy/PyTorch CPU RNG state
  + config and tokenizer snapshots
  + trainable policy, adapter, and generated-parameter metadata
  -> local full checkpoint artifact
```

Implemented behavior:

- `mopforge.lifecycle`: typed `TrainingCheckpointRecord`, RNG capture/restore,
  and torch-backed `save_full_training_checkpoint` /
  `load_full_training_checkpoint`.
- Artifact integration: `CheckpointManager.save_full_training_checkpoint`
  writes under `artifacts/checkpoints/`, registers a normal `checkpoint`
  artifact with `full_checkpoint: true`, and supports
  `latest_full_checkpoint(...)` by run ID/model type/training kind.
- `TrainerConfig` lifecycle fields:
  `save_full_checkpoints`, `resume_from_checkpoint`,
  `checkpoint_every_steps`, `save_rng_state`, `save_optimizer_state`, and
  `save_scheduler_state`.
- TinyTrainer resume: model state, optimizer state, trainer state/global step,
  and RNG state are restored when present and compatible. Scheduler state is
  represented but remains `None` unless a future scheduler is added.
- SFT resume: `FinetuneConfig` maps resume fields into TinyTrainer and tags
  checkpoints with `training_kind="sft"`.
- Continued pretraining resume: CPT saves and restores full checkpoints in its
  own tiny loop and continues `global_step`.
- CLI resume commands:

```bash
mopforge train resume artifacts/checkpoints/<checkpoint>.pt
mopforge sft resume artifacts/checkpoints/<checkpoint>.pt
mopforge pretrain resume artifacts/checkpoints/<checkpoint>.pt
mopforge train resume <run_id> --config configs/examples/tiny_trainer_mop_cpu.json
```

If a resume argument is a file path, it is used directly. Otherwise the CLI
looks for a full-checkpoint artifact ID or latest full checkpoint for that run
ID in the local artifact manifest. Without `--config`, the CLI uses the config
snapshot inside the checkpoint when possible and auto-extends `max_steps` by one
if the snapshot already reached its target step. With `--config`, the supplied
config is respected.

Run the lifecycle demo:

```bash
python examples/resume_training_demo.py
```

The demo performs one TinyTrainer, one SFT, and one CPT 1-step run, finds each
latest full checkpoint, resumes each for one additional step, and prints the
original run ID, checkpoint artifact/path, resumed run ID, step transition, and
result path.

## Goal 25: Experiment Registry + Matrix Runner MVP

Goal 25 adds local experiment orchestration:

```text
experiment config
  -> matrix/list expansion
  -> runnable config envelopes
  -> sequential local CPU runs
  -> experiment registry
  -> summary JSON/CSV
```

Implemented behavior:

- `ExperimentConfig`: supports `kind="matrix"` for Cartesian product expansion
  from a `base_config`, and `kind="list"` for explicit runnable config
  envelopes.
- Matrix keys use dotted payload/metadata paths such as `payload.mode` and
  `payload.max_steps`; expansion preserves config key order, adds experiment
  metadata to each child run, supports `max_runs`, and rejects invalid paths.
- `ExperimentRegistry`: writes local records under:

```text
experiments/<experiment_id>/
  experiment.json
  expanded_runs.json
  record.json
  summary.json
  summary.csv
  run_records/<index>.json
```

- `run_experiment`: executes child `sft`, `pretrain`, and `trainer` envelopes
  sequentially, catches per-run failures, continues by default, and records
  `completed`, `completed_with_failures`, or `failed` experiment status.
- Summary rows include experiment/run IDs, child kind, SFT mode, model type,
  policy mode, train/eval losses when available, finite flag, result path, and
  error text for failed children.
- Config support: `kind="experiment"` envelopes now validate, dry-run, and map
  to `ExperimentConfig`.
- Default templates:
  `experiment_dense_vs_mop` and `experiment_adapter_vs_generated`.
- CLI commands:

```bash
mopforge experiment dry-run configs/examples/experiment_adapter_vs_generated.json
mopforge experiment run configs/examples/experiment_adapter_vs_generated.json
mopforge experiment list
mopforge experiment show <experiment_id>
```

Tracked example configs:

```text
configs/examples/experiment_dense_vs_mop.json
configs/examples/experiment_adapter_vs_generated.json
```

Run the experiment example:

```bash
python examples/run_experiment_matrix.py
```

The example prepares tiny lessons/index if needed, runs the adapter-vs-generated
SFT experiment, and prints the experiment ID, run counts, child run IDs, and
summary JSON/CSV paths.

## Goal 26: Benchmark & Evaluation Suite MVP

Goal 26 adds local benchmark/evaluation plumbing:

```text
benchmark config
  -> evaluator
  -> metrics/examples
  -> benchmark registry
```

Implemented behavior:

- `BenchmarkConfig`: a CPU-only benchmark envelope payload with benchmark type,
  model/checkpoint/run references, KTS paths, tokenizer settings, generation
  limits, adapter/generated flags, output root, and metadata.
- `BenchmarkRegistry`: writes local benchmark records under:

```text
benchmarks/<benchmark_id>/
  benchmark.json
  metrics.json
  metrics.csv
  examples.json
  record.json
```

- Metric helpers: `safe_mean`, `safe_rate`, `count_by_key`, `flatten_metrics`,
  `finite_float`, and `json_safe`.
- Evaluators:
  - `loss`: tiny causal-LM loss over KTS lessons.
  - `code_correctness`: greedy generation, code extraction, verifier pass/fail,
    failures by type, and example previews.
  - `router`: untrained or checkpoint-loaded tiny router prediction smoke
    metrics, exact-match rate, average module counts, and per-module TP/FP/FN.
  - `parameter_efficiency`: total/trainable/frozen parameter counts, trainable
    ratio, and policy group summaries.
  - `composite`: tiny parameter-efficiency, loss, and code-correctness bundle.
- `run_benchmark`: creates a benchmark record, dispatches the evaluator, writes
  `metrics.json`, flattened one-row `metrics.csv`, optional `examples.json`,
  and captures evaluator failures in a failed benchmark record.
- Config support: `kind="benchmark"` envelopes validate, dry-run, and map to
  `BenchmarkConfig`.
- Default templates:
  `benchmark_loss`, `benchmark_code_correctness`, `benchmark_router`,
  `benchmark_parameter_efficiency`, and `benchmark_composite`.
- CLI commands:

```bash
mopforge benchmark dry-run configs/examples/benchmark_composite.json
mopforge benchmark run configs/examples/benchmark_composite.json
mopforge benchmark list
mopforge benchmark show <benchmark_id>
```

Tracked example configs:

```text
configs/examples/benchmark_loss.json
configs/examples/benchmark_code_correctness.json
configs/examples/benchmark_router.json
configs/examples/benchmark_parameter_efficiency.json
configs/examples/benchmark_composite.json
```

Run benchmark examples:

```bash
python examples/run_benchmarks.py
```

Benchmarks can be run separately after experiments by pointing a benchmark
config at a run ID or compatible checkpoint path. The MVP records those source
fields in metrics/records, but does not yet produce paper-style comparison
reports.

## Goal 27: Result Analysis + Comparison Reports MVP

Goal 27 adds local analysis/report plumbing:

```text
experiment summaries
+ benchmark metrics
+ run result JSON files
-> normalized rows
-> comparison tables
-> Markdown report + JSON/CSV artifacts
```

Implemented behavior:

- `AnalysisConfig`: a local report config with experiment IDs, benchmark IDs,
  run result paths, output root, metric selection, grouping, rank metric/mode,
  optional baseline filter, and metadata.
- `AnalysisRegistry`: writes local analysis records under:

```text
reports/<analysis_id>/
  analysis.json
  normalized_results.json
  normalized_results.csv
  comparison.json
  comparison.csv
  report.md
  record.json
```

- Loading helpers:
  - `load_experiment_summary`: reads `summary.json` by experiment ID, directory,
    or direct path, with `summary.csv` fallback.
  - `load_benchmark_metrics`: reads benchmark `metrics.json` by ID, directory,
    or direct path.
  - `load_run_result`: reads `trainer_result.json`, `finetune_result.json`,
    `continued_pretrain_result.json`, or `metrics.json`.
- Normalization helpers turn heterogeneous artifacts into stable rows with
  source, run, kind, mode, model/routing, trainable-policy, parameter-count,
  loss, pass-rate, router-rate, finite, result-path, error, and metadata fields.
- Comparison helpers support grouping, group summaries, min/max ranking,
  best-row selection, baseline filtering, and numeric deltas vs baseline.
- Markdown report generation includes sources, normalized row count, ranking
  table, group summaries, benchmark metric highlights, baseline deltas when
  configured, and limitations.
- `run_analysis`: creates a report record, loads sources, normalizes rows,
  compares results, writes JSON/CSV artifacts, writes `report.md`, and records
  failed analyses with error text.
- Config support: `kind="analysis"` envelopes validate, dry-run, and map to
  `AnalysisConfig`.
- Default templates:
  `analysis_adapter_vs_generated`, `analysis_dense_vs_mop`, and
  `analysis_composite_report`. These are source-empty templates by default,
  with metadata documenting where to add dynamic local IDs.
- CLI commands:

```bash
mopforge analyze experiment <experiment_id>
mopforge analyze benchmark <benchmark_id>
mopforge analyze compare --experiments <id1> <id2> --benchmarks <bid1> <bid2> --rank-by final_eval_loss
mopforge analyze list
mopforge analyze show <analysis_id>
mopforge report build configs/examples/analysis_composite_report.json
```

Tracked example configs:

```text
configs/examples/analysis_adapter_vs_generated.json
configs/examples/analysis_dense_vs_mop.json
configs/examples/analysis_composite_report.json
```

Run the analysis example:

```bash
python examples/analyze_results.py
```

The example reuses existing local experiment/benchmark outputs when available,
or creates tiny CPU-safe smoke artifacts first. It then writes normalized
results, comparison JSON/CSV, and a Markdown report. These are inspection
artifacts, not paper-quality analysis.

## Goal 28: Dataset Registry + Dataset Versioning MVP

Goal 28 adds local dataset reproducibility plumbing:

```text
local JSONL paths
-> fingerprints + stats
-> manifest/version
-> deterministic split refs
-> materialized split JSONL
```

Implemented behavior:

- `FileFingerprint`: records normalized path, size, sha256, and modified time
  without loading large files fully into memory.
- Dataset stats for `lessons`, `corpus`, `generic_jsonl`, and `split` JSONL
  files. Lesson stats count domains, skills, target modules, and verification
  status; corpus stats count sources, domains, and languages. Malformed lines
  are counted in stats metadata instead of stopping fingerprinting.
- `DatasetManifest`: records dataset ID, version ID, kind, source paths,
  fingerprints, combined sha256, stats, tags, description, and metadata.
- `DatasetRegistry`: writes local records under:

```text
datasets/
  registry.json
  <dataset_id>/
    dataset.json
    versions/
      <version_id>/
        manifest.json
        stats.json
        files/
        splits/
          <split_id>.json
```

- Versioning supports latest refs, explicit `dataset_id@version_id` refs, and
  direct `manifest.json` paths. Default mode references original files; optional
  copied-file mode stores source snapshots under `files/`.
- Deterministic train/eval/test splits with stable split IDs such as
  `split-seed123-train80-eval10-test10`. Lesson datasets use lesson IDs when
  available; generic records use line indices. Simple best-effort stratification
  by `skill`, `domain`, or `target_module` is supported.
- Materialization helpers load records for a split and write split JSONL files
  while preserving original JSON records.
- Config support: `kind="dataset"` envelopes validate, dry-run, and map to
  `DatasetConfig`.
- Optional config references were added to Trainer/SFT/CPT/Benchmark configs.
  Existing path-based behavior is preserved; dataset refs are recorded in
  metadata and benchmark metrics when resolvable.
- Default templates:
  `dataset_register_lessons`, `dataset_register_corpus`, and
  `dataset_split_lessons`.
- CLI commands:

```bash
mopforge dataset register data/coding_bugfix_lessons.jsonl --name coding_bugfix --kind lessons
mopforge dataset snapshot coding_bugfix
mopforge dataset split coding_bugfix --train 0.8 --eval 0.1 --test 0.1 --seed 123
mopforge dataset list
mopforge dataset show coding_bugfix
mopforge dataset show coding_bugfix@<version_id>
mopforge dataset versions coding_bugfix
mopforge dataset materialize-split coding_bugfix --split-id <split_id> --split train --output outputs/dataset_demo/train.jsonl
```

Tracked example configs:

```text
configs/examples/dataset_register_lessons.json
configs/examples/dataset_split_lessons.json
configs/examples/dataset_register_corpus.json
```

Run the dataset registry example:

```bash
python examples/manage_datasets.py
```

The example ensures tiny coding bugfix lessons exist, registers and snapshots a
local dataset, creates a deterministic split, materializes train/eval JSONL
files, and prints the dataset/version/fingerprint/stats/split paths.

## Goal 29: Model Registry + Architecture Configs MVP

Goal 29 adds local model-architecture metadata and versioned model records:

```text
architecture config
-> manifest + parameter summary
-> models/<model_id>/versions/<version_id>/
```

Implemented behavior:

- `ModelArchitectureConfig`: standard-library architecture metadata for
  `dense`, `mop_oracle`, `mop_learned_router`, `baseline_moe`, and
  `future_large` plans. Tiny dense/MoP architectures can instantiate local
  smoke-test models when PyTorch is available.
- `ModelManifest` and `ModelConfig`: versioned model metadata, tags,
  descriptions, tokenizer/config refs, optional checkpoint refs, and safe JSON
  round-trips.
- `ModelRegistry`: writes `manifest.json`, `architecture.json`,
  `parameter_summary.json`, per-model records, latest-version refs, explicit
  `model_id@version_id` resolution, and snapshot versions.
- The `baseline_moe` architecture is an explicit tiny MoE/MoP-compatible shim
  for comparison plumbing. `future_large` records validate and serialize, but
  refuse local instantiation.
- Config support: `kind="model"` envelopes validate, dry-run, and map to
  `ModelConfig`.
- Default templates:
  `model_tiny_dense`, `model_tiny_mop`, `model_tiny_adapter`,
  `model_tiny_generated`, and `model_future_2b_mop`.
- CLI commands:

```bash
mopforge model register configs/examples/model_tiny_mop.json
mopforge model list
mopforge model show <model_id>
mopforge model versions <model_id>
mopforge model snapshot <model_id>
```

Run the model registry example:

```bash
python examples/manage_models.py
```

## Goal 30: Research Run Manifests MVP

Goal 30 adds local launch-manifest planning without launching remote jobs:

```text
config envelope + resource spec
-> manifest
-> dry-run payload + command text
```

Implemented behavior:

- `ResourceSpec`: validates accelerator family, GPU count, memory estimate,
  precision, CPU threads, and notes. CPU specs use zero GPUs; GPU accelerator
  plans require at least one GPU.
- `ResearchRunManifest` and `ManifestConfig`: serializable run plans with
  config refs/payload snapshots, datasets/models/checkpoints, expected
  artifacts, command text, and reproducibility metadata.
- `ManifestRegistry`: writes manifests under `manifests/<manifest_id>/` with
  `manifest.json`, `command.txt`, `dry_run.json`, and `record.json`.
- Planner helpers create deterministic local command strings such as
  `mopforge sft run <config>`, `mopforge experiment run <config>`, and
  `mopforge report build <config>`.
- Config support: `kind="manifest"` envelopes validate and dry-run.
- Default templates:
  `manifest_cpu_smoke`, `manifest_a100_2b_plan`, `manifest_h100_mop_plan`, and
  `manifest_b300_multi_gpu_plan`.
- CLI commands:

```bash
mopforge manifest create configs/examples/sft_full_cpu.json --accelerator cpu
mopforge manifest dry-run <manifest_id>
mopforge manifest list
mopforge manifest show <manifest_id>
mopforge manifest export-command <manifest_id>
```

Run the manifest example:

```bash
python examples/create_run_manifests.py
```

## Goal 31: Local Result Importer MVP

Goal 31 adds local import plumbing for outputs produced elsewhere, while still
using local files only:

```text
source directory or result file
-> copied artifacts + fingerprints
-> normalized rows
-> import record
```

Implemented behavior:

- `ResultImportConfig`, `ResultImportRecord`, and `ResultImportRegistry`:
  import local result directories/files into `imports/<import_id>/`.
- Artifact detection for trainer, SFT, continued-pretraining, experiment,
  benchmark, analysis, ablation, statistics, and paper-report JSON/Markdown
  files.
- Optional artifact copying plus SHA-256 fingerprints, `manifest.json`,
  `normalized_results.json`, `normalized_results.csv`, and `record.json`.
- Result normalization reuses the Goal 27 analysis row schema where possible.
- Config support: `kind="import"` envelopes validate and dry-run.
- Default template: `import_results`.
- CLI commands:

```bash
mopforge import results runs --name local_runs_import
mopforge import list
mopforge import show <import_id>
```

Run the importer example:

```bash
python examples/import_results_demo.py
```

## Goal 32: Ablation Framework MVP

Goal 32 adds tiny sequential ablation plumbing:

```text
base config + variants
-> child experiment configs
-> experiment run
-> analysis report
```

Implemented behavior:

- `AblationVariant` and `AblationConfig`: named override sets with tags,
  metadata, rank metric/mode, group keys, source refs, and local output roots.
- `expand_ablation_variants`: applies per-variant payload/metadata overrides to
  a base config envelope.
- `run_ablation`: creates an ablation record, runs a sequential local
  experiment, runs analysis over the experiment summary, and writes
  `ablation.json`, `expanded_runs.json`, `summary.json`, `report.md`, and
  `record.json`.
- `dry_run_ablation`: reports variant count, output roots, rank mode, and child
  dry-run summaries without training.
- Config support: `kind="ablation"` envelopes validate, dry-run, and map to
  `AblationConfig`.
- Default templates:
  `ablation_adapter_vs_generated`, `ablation_dense_vs_mop`, and
  `ablation_fastparam_policy`.
- CLI commands:

```bash
mopforge ablation dry-run configs/examples/ablation_adapter_vs_generated.json
mopforge ablation run configs/examples/ablation_adapter_vs_generated.json
mopforge ablation list
mopforge ablation show <ablation_id>
```

Run the ablation example:

```bash
python examples/run_ablation.py
```

## Goal 33: Baseline Framework MVP

Goal 33 adds a named local baseline catalog for comparison configs:

- `BaselineSpec` and `BaselineConfig`: metadata for model family, training
  mode, parameter strategy, adapter/generated flags, tags, and descriptions.
- Catalog entries:
  `dense_full`, `dense_head_only`, `mop_oracle_full`, `mop_module_only`,
  `adapter_only`, `generated_params_only`, `mop_learned_router`, and
  `moe_tiny`.
- `moe_tiny` is intentionally labeled `moe_tiny_shim`: it is backed by local
  TinyMoP-compatible plumbing, not a new production MoE implementation.
- `build_baseline_experiment_config`: creates a tiny local list experiment from
  selected baseline names.
- Config support: `kind="baseline"` envelopes validate and dry-run.
- Default template: `baseline_dense_adapter_mop`.
- CLI commands:

```bash
mopforge baseline list
mopforge baseline show moe_tiny
mopforge baseline experiment --baselines dense_full adapter_only generated_params_only
```

Run the baseline example:

```bash
python examples/run_baseline_comparison.py
```

## Goal 34: Statistical Reporting Tables MVP

Goal 34 adds simple standard-library statistical summaries for normalized
result rows:

- `mean`, `median`, `stddev`, `stderr`, and `percent_change`.
- `summarize_metric`, `summarize_by_group`, and `compare_groups_simple`.
- `make_metric_table` plus JSON/CSV/Markdown table writers.
- CLI summary command over row JSON files:

```bash
mopforge stats summarize reports/<analysis_id>/normalized_results.json --group-by mode --metric final_eval_loss
```

Run the statistics example:

```bash
python examples/statistical_tables.py
```

The MVP deliberately does not implement significance testing, confidence
intervals, plotting, or paper-grade statistical claims.

## Goal 35: Paper-Style Report Scaffolds MVP

Goal 35 adds conservative Markdown scaffolding for future research reports:

```text
analysis/table/model/dataset refs
-> paper_reports/<paper_report_id>/report.md
```

Implemented behavior:

- `PaperReportConfig`, `PaperReportRecord`, and `PaperReportRegistry`.
- `build_paper_report`: writes a structured Markdown report with title,
  abstract, claim-status notice, methods, datasets, models, experiments,
  results, limitations, reproducibility checklist, and appendix refs.
- Config support: `kind="paper_report"` envelopes validate, dry-run, and map to
  `PaperReportConfig`.
- Default template: `paper_report_smoke`.
- CLI commands:

```bash
mopforge paper build configs/examples/paper_report_smoke.json
mopforge paper list
mopforge paper show <paper_report_id>
```

Run the paper scaffold example:

```bash
python examples/build_paper_report.py
```

Paper reports are Markdown scaffolds only. They are not PDFs, LaTeX exports,
peer-reviewed claims, or automatic paper-quality analyses.

## Goal 36: GPU / Device Runtime Foundation MVP

Goal 36 adds the first device and precision runtime layer without turning
MoP-Forge into a distributed or production GPU trainer:

```text
runtime config
-> device detection
-> precision policy
-> runtime context
-> trainer / SFT / CPT / benchmark metadata
```

Implemented behavior:

- `mopforge.runtime`: `RuntimeConfig`, `DeviceInfo`, `PrecisionPolicy`, and
  `RuntimeContext`.
- Device detection for CPU, optional CUDA inventory, optional MPS availability,
  PyTorch version, CUDA version, GPU names, memory, and capabilities when
  available.
- Device resolution for `cpu`, `auto`, `cuda`, `cuda:N`, and `mps`, including
  CPU fallback when `require_device_available=False`.
- Precision planning for `fp32`, `fp16`, `bf16`, `auto`, and planning-only
  `fp8`, plus best-effort TF32 backend toggles.
- Runtime context helpers for moving models and nested batches, applying
  deterministic settings, entering autocast only when safe, and recording
  runtime warnings.
- TinyTrainer, FT/SFT, continued pretraining, and benchmark evaluators now
  accept runtime fields and write runtime metadata into result/checkpoint
  artifacts where applicable.
- Config support: `kind="runtime"` envelopes validate and dry-run, and trainer,
  SFT, pretrain, and benchmark configs can include runtime fields.
- Default templates: `runtime_cpu`, `runtime_auto`,
  `runtime_cuda_bf16_plan`, `trainer_runtime_auto`, `sft_runtime_auto`, and
  `benchmark_runtime_auto`.
- CLI commands:

```bash
mopforge runtime detect
mopforge runtime dry-run --device cpu --precision fp32
mopforge runtime dry-run --device auto --precision auto
mopforge runtime dry-run --device cuda --precision bf16 --no-require-available
```

Run the runtime examples:

```bash
python examples/runtime_detection.py
python examples/train_tiny_runtime_cpu.py
python examples/run_runtime_config_smoke.py
```

This is a foundation layer. CPU remains the default and required path. CUDA is
optional and smoke-level only when available; there is still no distributed
execution, FSDP, DeepSpeed, sharded checkpointing, production GradScaler loop,
GPU job launcher, large-model training, or GPU-efficient MoP routing.

## Goal 37: GPU Trainer MVP

Goal 37 adds `mopforge.gpu` and a `GPUTrainer` that builds on the Goal 36
runtime layer:

- `GPUTrainingConfig`, `GPUTrainingState`, and `GPUTrainingResult`.
- Tiny dense/MoP construction from inline settings or model refs.
- Runtime-aware model and batch movement.
- Trainable-parameter policies reused from the CPU trainer.
- Local `gpu_runs/<run_id>/` outputs: `config.json`, `runtime.json`,
  `metrics.json`, `state.json`, `gpu_training_result.json`, memory estimates,
  and checkpoints.
- CPU fallback remains supported for tests and development.

## Goal 38: AMP / GradScaler / Gradient Accumulation / Activation Checkpointing

Implemented behavior:

- `AmpScaler` enables GradScaler only for CUDA fp16 AMP.
- bf16, fp32, and CPU paths run without a scaler.
- Gradient accumulation divides loss and reports global steps, optimizer steps,
  samples seen, tokens seen, micro batch size, accumulation steps, and effective
  batch size.
- Activation checkpointing is recorded as an explicit hook with metadata and
  warnings for tiny models; it does not rewrite the tiny model graph yet.
- Efficient attention selection records `torch_sdpa` when available, otherwise
  eager fallback.

## Goal 39: GPU Data Pipeline / Streaming Loader

Implemented behavior:

- `GPUDataConfig` for lesson/corpus/dataset-ref loading.
- JSONL lesson loading through existing KTS records.
- Corpus loading through the continued-pretraining corpus API.
- Dataset registry refs and split materialization are supported when split JSON
  metadata exists locally.
- `StreamingJSONLDataset` provides a tiny iterable JSONL path.
- DataLoader construction uses pinned memory only when CUDA is selected.
- Data metadata records source refs, record counts, max sequence length, worker
  count, and pin-memory decisions.

## Goal 40: GPU Checkpointing + Resume

Implemented behavior:

- GPUTrainer checkpoints include model, optimizer, scheduler, scaler, runtime
  metadata, trainer state, config, RNG state, data/model/memory metadata, global
  step, optimizer step, and tokens seen.
- CPU RNG and CUDA RNG capture/restore reuse the local lifecycle helpers.
- `mopforge gpu resume <checkpoint_or_run_id>` resumes from a checkpoint path
  or latest local GPU run checkpoint.
- Checkpoints are local filesystem artifacts, not distributed sharded
  checkpoints.

## Goal 41: Single-GPU Serious Job Profiles

Tracked job profiles:

```text
configs/jobs/tiny_gpu_smoke.json
configs/jobs/100m_dense_a100_smoke.json
configs/jobs/100m_mop_a100_smoke.json
configs/jobs/500m_dense_vs_mop_h100.json
configs/jobs/1b_mop_h100_bf16.json
configs/jobs/2b_mop_a100_plan.json
configs/jobs/7b_mop_h100_plan.json
```

`estimate_training_memory` provides approximate planning estimates for weights,
gradients, AdamW state, and activations. Large profiles are validation/planning
records unless explicitly run by the user on suitable hardware and data.

## Goal 42: Torchrun / Multi-GPU Launcher Foundation

Implemented behavior:

- `DistributedConfig` validates torchrun-style launch metadata.
- `build_torchrun_command` and `mopforge gpu launch-torchrun ... --dry-run`
  print the planned command without executing it.
- `configs/jobs/multigpu_mop_torchrun_plan.json` records a dry-run torchrun
  plan.

This is a launcher foundation only; multi-GPU training is not production
hardened.

## Goal 43: GPU-Efficient MoP Routing + Fast Parameters

Implemented behavior:

- `ModuleRoutingPlan`, routing-density helpers, and grouping metadata.
- Active-parameter estimates for MoP runs.
- Fast-adapter and generated-condition density metadata.
- GPUTrainer metrics include routing mode, active module density, active adapter
  density, generated condition density, trainable parameter ratio, and active
  parameter estimates when available.

This remains PyTorch-level metadata/grouping support, not custom fused CUDA
kernels.

## Goal 44: Serious Job CLI + Documentation Polish

Implemented CLI:

```bash
mopforge gpu validate <config>
mopforge gpu estimate <config>
mopforge gpu train <config>
mopforge gpu resume <checkpoint_or_run_id>
mopforge gpu benchmark <run_id>
mopforge gpu launch-torchrun <config> --dry-run
mopforge gpu list
mopforge gpu show <run_id>
```

Additional docs:

```text
docs/gpu_quickstart.md
docs/gpu_job_profiles.md
docs/gpu_runtime_limitations.md
docs/serious_jobs_checklist.md
```

MoP-Forge now includes a serious single-GPU research beta for tiny-to-small MoP
experiments and validated large-job profiles. It is not yet a fully production
distributed LLM training framework.

## Goal 46: GPU Efficiency Benchmarking

Implemented behavior:

- GPU training writes nested `metrics.efficiency` data into `metrics.json` and
  `gpu_training_result.json`, including throughput, step timing, CUDA memory,
  trainable/frozen parameter counts, active-parameter estimates, routing
  densities, and checkpoint size.
- CUDA memory tracking uses PyTorch peak/current allocated and reserved memory
  APIs when CUDA is selected, and records `null` safely on CPU/no-CUDA runs.
- MoP trainable policies now include full training, adapters-only,
  modules-only, core-frozen, and router/adapters-only modes with parameter
  group summaries showing frozen and trainable groups.
- Colab-safe 100M dense/MoP efficiency configs live under `configs/jobs/`.
- `mopforge gpu compare-runs` and `scripts/compare_gpu_runs.py` compare old
  and new GPU run artifacts and emit readable tables plus JSON/CSV outputs.

## Examples

Create demo lessons:

```bash
python examples/create_lessons.py
```

Read and filter verified coding/debugging lessons:

```bash
python examples/read_and_filter_lessons.py
```

After running the Goal 2 generator, the reader uses
`data/coding_bugfix_lessons.jsonl`; otherwise it falls back to the smaller Goal
1 demo file.

Tokenize lessons for causal-LM training:

```bash
python examples/tokenize_lessons.py
```

Smoke-test the tiny dense baseline if PyTorch is installed:

```bash
python examples/train_tiny_dense_baseline.py
```

Smoke-test the tiny oracle-routed MoP baseline if PyTorch is installed:

```bash
python examples/train_tiny_mop_baseline.py
```

Smoke-test the tiny learned router if PyTorch is installed:

```bash
python examples/train_tiny_router.py
```

Smoke-test TinyMoP with learned-router predictions:

```bash
python examples/train_tiny_mop_with_learned_router.py
```

Run the dense vs MoP comparison harness:

```bash
python examples/run_tiny_comparison.py
```

Run generated-code evaluation:

```bash
python examples/evaluate_tiny_generated_code.py
```

Build repair lessons from generated-code failures:

```bash
python examples/build_repair_lessons_from_tiny_eval.py
```

Build the SQLite KTS metadata index:

```bash
python examples/index_kts_lessons.py
```

Build curriculum plans from the indexed KTS:

```bash
python examples/schedule_curriculum.py
```

Run tiny curriculum-driven training records:

```bash
python examples/run_curriculum_training.py
```

Build feedback-aware curriculum records:

```bash
python examples/feedback_aware_curriculum.py
```

Run one feedback-weighted retraining loop:

```bash
python examples/run_feedback_retraining_loop.py
```

Build and consume one local module training queue item:

```bash
python examples/build_module_training_queue.py
```

Manage local artifacts and tiny checkpoints:

```bash
python examples/manage_artifacts_and_checkpoints.py
```

Run the CPU-first tiny trainer skeleton:

```bash
python examples/run_tiny_trainer.py
```

Run module-specific trainable policy smoke tests:

```bash
python examples/train_module_specific_policy.py
```

Run fast-adapter smoke training:

```bash
python examples/train_fast_adapter.py
```

Run FT/SFT mode API smoke examples:

```bash
python examples/run_sft_modes.py
```

Run continued-pretraining corpus API smoke training:

```bash
python examples/run_continued_pretraining.py
```

Run tokenizer abstraction demo:

```bash
python examples/tokenizer_abstraction_demo.py
```

Run generated-parameter smoke training:

```bash
python examples/train_generated_params.py
```

Run config/CLI workflow demo:

```bash
python examples/run_cli_configs.py
```

Run full checkpoint resume demo:

```bash
python examples/resume_training_demo.py
```

Run local experiment matrix/list demo:

```bash
python examples/run_experiment_matrix.py
```

Run local benchmark suite demo:

```bash
python examples/run_benchmarks.py
```

Build a local analysis report:

```bash
python examples/analyze_results.py
```

Manage local dataset versions and splits:

```bash
python examples/manage_datasets.py
```

Manage local model manifests and architecture snapshots:

```bash
python examples/manage_models.py
```

Create local research run manifests:

```bash
python examples/create_run_manifests.py
```

Import local or externally-produced result artifacts:

```bash
python examples/import_results_demo.py
```

Run a tiny CPU ablation:

```bash
python examples/run_ablation.py
```

Build a tiny baseline comparison config:

```bash
python examples/run_baseline_comparison.py
```

Write statistical summary tables:

```bash
python examples/statistical_tables.py
```

Build a paper-style Markdown report scaffold:

```bash
python examples/build_paper_report.py
```

Inspect runtime/device availability and run CPU-safe runtime smoke configs:

```bash
python examples/runtime_detection.py
python examples/train_tiny_runtime_cpu.py
python examples/run_runtime_config_smoke.py
```

Run GPU beta examples with CPU fallback where needed:

```bash
python examples/gpu_train_tiny_smoke.py
python examples/gpu_gradient_accumulation_demo.py
python examples/gpu_memory_estimate.py
python examples/gpu_resume_demo.py
python examples/gpu_mop_routing_demo.py
python examples/gpu_job_profile_validate.py
python examples/gpu_torchrun_dry_run.py
```

## Tests

```bash
python -m pytest -q
```

## Verifier Warning

`mopforge.verify.verify_python_solution` is not a secure sandbox. It writes the
candidate and test code to a temporary file and runs it with the local Python
interpreter. Only run trusted code locally. Docker or another real sandbox is a
future milestone and is intentionally not implemented here.

## Current Limitations

- JSONL remains canonical; SQLite is a metadata index, not a replacement.
- No true concurrent write safety beyond SQLite basics.
- JSONL and SQLite can get out of sync if files are manually edited; rebuild
  the index from JSONL.
- No vector search.
- No production database server.
- Curriculum scheduler is simple and deterministic.
- Feedback-aware scheduling is offline, not live adaptive training.
- Feedback scoring is heuristic and inspectable, not learned.
- Tiny model failures are noisy and should not be treated as real quality
  signals.
- No reinforcement-learning curriculum.
- Repair boosting is heuristic and does not duplicate IDs.
- Local module queues exist, but there are no production workers.
- No distributed queue, async execution, or background worker system.
- No production feedback-retraining queue system.
- Curriculum training runner is not production training.
- Feedback retraining loop is a tiny smoke loop, not a long-running trainer.
- Local checkpoint artifacts exist, but there is no production checkpoint
  retention policy.
- No remote artifact store, model hub integration, or checkpoint publishing.
- Full checkpoint resume is local filesystem only.
- Optimizer state is saved/restored for local tiny runs when compatible.
- Scheduler lifecycle is a minimal nullable slot; no real scheduler policy is
  implemented yet.
- Compatibility is best-effort if config/model architecture changes between
  save and resume.
- No distributed checkpointing.
- No sharded checkpoints.
- CUDA RNG restore is not production-ready; CPU RNG state remains the reliable
  implemented path.
- Trainable-parameter grouping is name-based and tiny-model-specific.
- Target module isolation is a local MVP, not a proven training method.
- Module-specific policy runs do not make real module-quality claims.
- Fast adapters are tiny local modules, not generated dynamically.
- Adapter routing is simple and heuristic.
- Generated parameters are a tiny hypernetwork MVP only.
- Generated tensors are per-forward adapter tensors, not persisted standalone
  artifacts.
- Generated parameters are not large dynamic production weights.
- No advanced generated-parameter routing or gating yet.
- No large hypernetwork support.
- No large adapter training.
- YAML config files require optional PyYAML; JSON configs work with the
  standard library.
- CLI commands are local smoke runners and runtime planners, not production
  launchers.
- Distributed CLI support is torchrun dry-run planning only; it does not launch
  a production multi-GPU job.
- CLI runtime fields can resolve optional single-device CUDA/MPS requests when
  the local PyTorch install supports them, but there is no GPU job launcher.
- Experiment runner is sequential local smoke execution only; optional device
  runtime fields are resolved inside child configs.
- No parallel experiment workers.
- No distributed experiment jobs.
- GPUTrainer can execute local single-device tiny/small jobs, but research run
  manifests still do not launch GPU jobs.
- Model registry records local metadata and tiny architecture summaries; it is
  not a model hub.
- `future_large` model configs are planning records only and do not instantiate
  local large models.
- `baseline_moe`/`moe_tiny` is a tiny shim for comparison plumbing, not a
  production MoE implementation.
- Research run manifests are planning artifacts only; they do not allocate GPUs,
  submit jobs, or run remote/cloud execution.
- GPU accelerator specs are allowed in manifests as future plans. GPUTrainer
  can run local jobs from `gpu_train` configs, but manifests still do not
  launch remote/cloud GPU jobs.
- Result importer reads local files and directories only; there is no SSH,
  object-store, experiment-tracker, or remote registry integration.
- Ablation runner is sequential local smoke plumbing.
- Baseline comparison configs are tiny smoke-test configs, not benchmark
  authority.
- Statistical tables are descriptive summaries only.
- No confidence intervals, significance tests, bootstrap tests, or hypothesis
  testing yet.
- No plotting dependency is required or implemented.
- Paper report generation is Markdown scaffolding only.
- No PDF, LaTeX, bibliography manager, or automatic publication-ready table
  generation yet.
- Paper reports do not turn tiny smoke metrics into research claims.
- Conditional matrix logic is minimal; use explicit list experiments when child
  configs need different required flags.
- Benchmark suite is CPU-smoke local plumbing, not a public benchmark suite.
- No external benchmark datasets or internet/download requirement.
- No paper-quality benchmark report generation yet.
- Generated-code benchmark results are tiny-model smoke signals, not meaningful
  quality claims.
- Analysis reports are local filesystem inspection artifacts, not paper-quality
  reports.
- No statistical significance testing or confidence intervals in analysis yet.
- No plotting dependency is required or implemented.
- No PDF report generation.
- Analysis metrics are only as meaningful as the source experiments and
  benchmarks.
- Analysis loading is local filesystem only; no remote result importers yet.
- Dataset registry is local filesystem only.
- No external dataset downloads.
- No Hugging Face datasets integration yet.
- No web-scale streaming or large ingestion pipeline.
- No dataset dedup/filter pipeline yet.
- Split stratification is simple and best-effort.
- No remote dataset registry.
- No immutable storage guarantees beyond local files and checksums.
- No production config schema registry yet.
- No large-scale SFT or continued pretraining.
- No streaming or web-scale corpus ingestion.
- No RLHF, DPO, or preference tuning.
- Router SFT is an MVP path over the existing tiny learned-router smoke flow,
  not a production router trainer.
- Optional Hugging Face/tokenizers compatibility exists only for tokenizers;
  there is no Hugging Face model integration yet.
- No tokenizer training yet.
- No mandatory Hugging Face dependency.
- No internet or download requirement for tokenizer tests/examples.
- Byte tokenizer remains the default for CPU smoke tests.
- No real queue concurrency safety beyond basic local SQLite behavior.
- No adaptive curriculum from live model metrics yet.
- No live online adaptation.
- No statistically meaningful before/after comparison.
- No meaningful quality claims.
- Runtime-aware TinyTrainer/SFT/CPT/benchmark paths can resolve CPU, auto,
  CUDA, CUDA index, or MPS requests, and GPUTrainer adds a single-device beta
  training loop. CPU remains the required default path.
- GPUTrainer includes an AMP scaler wrapper for CUDA fp16 and accumulation
  metadata, but it is not a production mixed-precision training system.
- FP8 is planning/fallback metadata only, not an execution precision.
- No FSDP, DeepSpeed, tensor parallelism, production multi-GPU training, custom
  efficient-attention kernels, or sharded GPU checkpoint pipeline yet.
- Torchrun support is a dry-run launcher foundation only.
- Activation checkpointing is currently hook/metadata support for tiny models,
  not a broad graph-rewrite system.
- Fast adapters, generated parameters, MoP routing, and MoE baselines have
  PyTorch-level metadata/grouping support but are not custom-kernel optimized.
- Generated-code eval in runs is optional and tiny.
- Feedback loop appends records, but does not automatically schedule repeated
  production retraining.
- HF/BPE tokenizer compatibility is an optional wrapper path, not a production
  tokenizer pipeline.
- Tiny dense model is only a smoke-test baseline.
- Oracle routing remains available for TinyMoP.
- Tiny learned router is experimental and CPU-smoke only.
- No serious router accuracy claims.
- Tiny comparison harness is not a benchmark.
- Generated-code evaluation uses greedy decoding only.
- Code extraction is intentionally simple.
- Pass rates are not meaningful yet.
- Repair lessons are generated from tiny model failures and are not yet
  high-quality curriculum data.
- No automatic retraining on repair lessons yet.
- No iterative repair agent yet.
- No production code benchmark.
- No large checkpoint testing.
- No guaranteed large-scale MoP training; production distributed training is
  still future work.
- No CUDA requirement; examples are designed to pass as CPU smoke tests, with
  CUDA paths optional when available locally.
- No Docker sandbox or security boundary for executing untrusted code.
- No cloud, database server, vector database, or web UI.
