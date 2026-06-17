# MoP-Forge Goal 45 Codex Prompt — v1.0-beta Hardening, Docs, API Cleanup, and Release Polish

You are Codex GPT-5.5 xHigh acting as the implementation agent for the MoP-Forge repository.

Implement Goal 45: v1.0-beta Hardening, Docs, API Cleanup, and Release Polish.

Target version after this goal:

```text
0.45.0
```

This goal is **not** about adding new research features.

This goal is about making the existing v0.44.0 serious GPU research beta clean, coherent, documented, testable, and ready for a public/research v1.0-beta release candidate.

---

# Read This First

Before changing code:

1. Read `README.md` fully.
2. Inspect the package layout:
   - `mopforge/kts/`
   - `mopforge/data/`
   - `mopforge/models/`
   - `mopforge/trainers/`
   - `mopforge/runtime/`
   - `mopforge/gpu/`
   - `mopforge/configs/`
   - `mopforge/cli/main.py`
   - `mopforge/datasets/`
   - `mopforge/experiments/`
   - `mopforge/benchmarks/`
   - `mopforge/analysis/`
   - `mopforge/papers/`
   - `mopforge/artifacts/`
   - `mopforge/lifecycle/`
3. Inspect all existing examples under `examples/`.
4. Inspect all config templates under:
   - `configs/examples/`
   - `configs/jobs/`
5. Inspect all docs under `docs/`.
6. Run the full test suite before making changes if practical:

```bash
python -m pytest -q
```

Use the results to avoid breaking the working v0.44 stack.

---

# Current Project State

MoP-Forge is now a serious single-GPU research beta at v0.44.0.

It includes:

- Knowledge Training Store
- verified coding/debugging lesson generation
- tiny dense and TinyMoP models
- learned router MVP
- generated-code evaluation
- repair loop
- SQLite KTS index
- curriculum scheduler
- feedback-aware curriculum
- feedback retraining loop
- module-specific training queue
- artifact/checkpoint manager
- CPU TinyTrainer
- trainable parameter policies
- fast adapters
- SFT modes
- continued-pretraining corpus API
- tokenizer abstraction
- generated parameters / hypernetwork MVP
- config/CLI system
- full local checkpoint resume
- experiment registry
- benchmark suite
- analysis reports
- dataset registry/versioning
- model registry
- run manifests
- result importer
- ablation framework
- baseline catalog
- statistics tables
- paper-style reports
- runtime/device/precision foundation
- GPUTrainingConfig and GPUTrainer beta
- AMP scaler wrapper
- gradient accumulation
- checkpoint/resume for GPU runs
- memory estimates
- job profiles
- torchrun dry-run planning
- MoP/Fast-Parameter routing metadata
- `mopforge gpu` CLI

Known limitations remain intentional:

- no production FSDP/DeepSpeed
- no production multi-GPU training
- no custom CUDA kernels
- no sharded distributed checkpoints
- no guaranteed 2B/7B training
- FP8 is planning-only
- torchrun support is dry-run foundation
- MoP routing is PyTorch-level metadata/grouping, not fused-kernel optimized
- docs are growing but need polish
- APIs have expanded rapidly and need cleanup
- config templates/examples need consistency

Goal 45 should polish and harden the project without overclaiming.

---

# Goal 45 Objective

Turn v0.44.0 into a coherent v0.45.0 release candidate for public research use.

The output should feel like:

```text
MoP-Forge v0.45.0:
clean serious GPU research beta
clear docs
consistent CLI
validated configs
stable public API surface
known limitations documented
ready for first A100/H100 experiments
```

Not:

```text
production distributed LLM trainer
finished MoP research result
commercial platform
```

---

# Hard Scope Boundaries

## Do not add major new ML features

Do not add:

- FSDP implementation
- DeepSpeed integration
- real distributed DDP training
- custom CUDA kernels
- new large model architecture
- web UI
- cloud launcher
- remote object store
- RLHF/DPO
- external dataset downloads
- benchmark dataset downloads

## Do polish and hardening

Focus on:

- docs
- examples
- config consistency
- CLI UX
- error messages
- validation
- API exports
- package structure
- smoke scripts
- release checklist
- test coverage for polish regressions

---

# Major Workstreams

## 1. Public API Audit

Create a clear public API policy.

Add:

```text
mopforge/public_api.py
```

or an equivalent module that documents/exports stable public symbols.

Audit `mopforge/__init__.py` and subpackage `__init__.py` files.

Requirements:

- Expose important public classes/functions consistently.
- Avoid dumping every internal helper into top-level namespace.
- Keep backwards compatibility with existing imports.
- Add comments for stable vs experimental APIs.
- Ensure `from mopforge import __version__` works.
- Ensure `mopforge version` works.
- Ensure optional torch imports do not break package import.

Add docs:

```text
docs/api_overview.md
```

Must include:

- stable public API
- experimental API
- internal/private modules
- compatibility policy

Tests:

- import top-level package
- import key subpackages
- import public API symbols
- verify optional torch absence behavior if project already supports it

---

## 2. CLI UX Cleanup

Audit `mopforge --help` and all major command groups.

Existing command groups include:

```text
version
modes
config
train
sft
pretrain
experiment
benchmark
analyze
report
dataset
model
manifest
import
ablation
baseline
stats
paper
runtime
gpu
```

Requirements:

- Ensure every command has clear help text.
- Ensure error messages are actionable.
- Ensure common typos/invalid config paths fail clearly.
- Ensure JSON-like outputs are readable and valid when intended.
- Ensure commands do not crash with tracebacks for user errors unless `--debug` exists.
- Add `--debug` or improve exception formatting if appropriate.
- Add consistent `--root`, `--output-root`, or document defaults.
- Ensure `mopforge gpu validate` clearly distinguishes validation, dry-run, and execution.
- Ensure `mopforge gpu train` warns if CPU fallback is happening.
- Ensure `mopforge gpu launch-torchrun --dry-run` never launches.

Add optional command:

```bash
mopforge doctor
```

If implemented, it should inspect:

- Python version
- package version
- PyTorch installed/version
- CUDA availability
- MPS availability
- writable output directories
- basic config examples present
- optional dependency status

If not implemented, document why.

Tests:

- CLI help for major groups
- invalid path error is clean
- invalid config kind error is clean
- `mopforge doctor` if implemented
- no CUDA required

---

## 3. Config Template Audit

Audit all default config templates.

Locations:

```text
configs/examples/
configs/jobs/
```

Required outcomes:

- Every config template validates with `mopforge config validate` or its specialized validator.
- Every config template dry-runs where applicable.
- Names are consistent.
- Paths are consistent and relative.
- CPU-safe configs are clearly separated from GPU job profiles.
- Large GPU configs are clearly marked as plans unless run on proper hardware.
- No template accidentally triggers large training in examples/tests.

Add docs:

```text
docs/config_templates.md
```

Include a table:

```text
config name | kind | purpose | CPU-safe? | GPU required? | executes training? | notes
```

Add tests:

- iterate over `configs/examples/*.json`
- iterate over `configs/jobs/*.json`
- validate/dry-run each appropriately
- skip execution for serious GPU configs

---

## 4. Example Audit and Repair

Audit every example under `examples/`.

Requirements:

- Examples should not fail due to stale duplicate records/indexes.
- Examples should create needed directories.
- Examples should be idempotent where practical.
- Examples should use temp/demo-safe output paths where possible.
- Examples should print concise useful results.
- Examples should not require CUDA unless explicitly optional and guarded.
- GPU examples should say whether they ran CPU fallback or CUDA.
- Long examples should stay tiny.

Add a script:

```text
scripts/run_smoke_examples.py
```

It should run a curated CPU-safe subset of examples and print pass/fail summary.

Optional flags:

```bash
python scripts/run_smoke_examples.py --quick
python scripts/run_smoke_examples.py --include-gpu-fallback
python scripts/run_smoke_examples.py --list
```

Add docs:

```text
docs/examples_guide.md
```

Tests:

- script lists examples
- quick mode runs a tiny subset
- no CUDA required

---

## 5. Documentation Restructure

The README is now very long. Keep it useful, but add structured docs.

Add or improve:

```text
docs/README.md
docs/architecture.md
docs/quickstart.md
docs/gpu_quickstart.md
docs/gpu_job_profiles.md
docs/gpu_runtime_limitations.md
docs/serious_jobs_checklist.md
docs/config_templates.md
docs/examples_guide.md
docs/api_overview.md
docs/release_checklist.md
docs/known_limitations.md
docs/research_positioning.md
```

README should become more navigable:

- top summary
- quick install
- quick CPU demo
- quick GPU beta demo
- architecture overview
- major capabilities
- commands overview
- docs index
- limitations summary

Do not delete detailed goal history unless replaced by docs. If README remains long, add a clear table of contents and links.

Docs must be truthful:

- serious single-GPU research beta
- not production distributed LLM trainer
- not guaranteed 2B/7B training
- no custom kernels
- no FSDP/DeepSpeed yet
- CPU fallback for tests does not validate GPU performance

---

## 6. GPU Release Readiness Checklist

Add:

```text
docs/serious_jobs_checklist.md
```

It must include the exact recommended first real hardware sequence:

```text
1. runtime detect
2. tiny GPU smoke
3. 100M dense
4. 100M MoP
5. dense vs MoP benchmark
6. 500M dense vs MoP
7. validate 1B
8. validate 2B only after 100M/500M are stable
```

Include commands:

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu train configs/jobs/100m_mop_a100_smoke.json
mopforge gpu benchmark <run_id>
mopforge gpu show <run_id>
```

Add expected outputs and troubleshooting.

---

## 7. Error Message and Validation Hardening

Improve validation errors across:

- config loading
- dataset refs
- model refs
- GPU job configs
- runtime device resolution
- checkpoint resume
- benchmark source lookup
- analysis source lookup

Requirements:

- Errors should say what failed.
- Errors should suggest how to fix it.
- Errors should include path/ref/kind when relevant.
- Invalid JSON/YAML should show path.
- CUDA unavailable should explain CPU fallback vs require behavior.
- Missing checkpoint/run ID should suggest `mopforge gpu list` or relevant list command.

Tests:

- invalid config path
- invalid dataset ref
- invalid model ref
- invalid GPU config
- unavailable CUDA required path
- missing resume checkpoint

---

## 8. Release Metadata and Versioning

Update version to:

```text
0.45.0
```

Ensure all version locations match:

- `mopforge/__init__.py`
- `pyproject.toml`
- any package metadata
- CLI version output

Add:

```text
CHANGELOG.md
```

If already exists, update it.

Changelog should include:

- v0.44.0 summary
- v0.45.0 hardening summary
- known limitations
- upgrade notes

Add:

```text
RELEASE_NOTES_v0.45.0.md
```

Include:

- what is new
- what is stable
- what is experimental
- how to run quick demos
- first GPU experiment sequence
- limitations

---

## 9. Packaging Polish

Audit `pyproject.toml`.

Check:

- project name
- version
- description
- readme
- Python version requirement
- dependencies
- optional dev dependencies
- console script entrypoint
- package discovery
- license field if present
- classifiers if present

Do not add heavy new dependencies.

Optional extras should be explicit if already used:

```text
dev
torch
hf
gpu
```

Only add extras if it does not break installation.

Add docs:

```text
docs/installation.md
```

Include:

- CPU install
- dev install
- optional tokenizer/HF install
- CUDA PyTorch note without hardcoding fragile install URLs
- Windows notes if relevant

---

## 10. Repository Hygiene

Add or improve:

```text
.gitignore
```

Ensure generated runtime outputs are ignored:

```text
runs/
gpu_runs/
artifacts/
outputs/
reports/
benchmarks/
experiments/
ablations/
paper_reports/
datasets/
models/
imports/
manifests/
data/*.jsonl
data/*.sqlite
__pycache__/
.pytest_cache/
```

Be careful not to ignore tracked config/example files.

Add:

```text
CONTRIBUTING.md
```

Include:

- run tests
- run smoke examples
- no large generated files in commits
- CPU-safe requirement
- optional CUDA tests
- style expectations

Optional:

```text
SECURITY.md
```

Include verifier warning: local Python verifier is not sandboxed.

---

## 11. Quality Gates Script

Add:

```text
scripts/release_check.py
```

It should run lightweight checks:

- import package
- check version consistency
- check CLI version
- validate selected configs
- run `mopforge runtime detect`
- verify docs files exist
- optionally run quick smoke examples

CLI:

```bash
python scripts/release_check.py
python scripts/release_check.py --quick-examples
python scripts/release_check.py --json outputs/release_check.json
```

Tests:

- release_check imports and basic checks pass
- JSON output written

---

## 12. Architecture Diagram / Text Diagram

Add a text-based architecture overview in:

```text
docs/architecture.md
```

Include diagrams like:

```text
KnowledgeLesson / Corpus
    -> Dataset Registry
    -> Tokenizer
    -> Trainer / GPUTrainer
    -> Model Registry / MoP Model
    -> Checkpoints / Artifacts
    -> Benchmark
    -> Analysis
    -> Paper Report
```

MoP parameter family diagram:

```text
Stable Core
+ Module Parameters
+ Router Parameters
+ Fast Adapters
+ Generated Parameters
+ Feedback/Curriculum Control
```

Clarify:

- lessons are not neural weights
- dataset/database stores training signals and provenance
- actual weights live in model/checkpoints/artifacts
- fast parameters currently include fast adapters and generated adapter tensors

---

## 13. v1.0-beta Positioning

Add:

```text
docs/research_positioning.md
```

Must answer:

- What is MoP-Forge?
- What problem is it trying to explore?
- How is MoP different from MoE?
- What does the framework currently prove?
- What does it not prove yet?
- What would count as successful first GPU evidence?

Do not overclaim.

Suggested positioning:

```text
MoP-Forge is a local-first research framework for testing Mixture-of-Parameters training ideas. It provides reproducible data/model/run/benchmark/report infrastructure and a serious single-GPU research beta, but it has not yet demonstrated large-scale MoP superiority.
```

---

## 14. Public Command Cookbook

Add:

```text
docs/command_cookbook.md
```

Include command groups:

- create lessons
- register dataset
- register model
- run CPU trainer
- run SFT
- run benchmark
- run analysis
- build paper report
- detect runtime
- validate GPU config
- run tiny GPU smoke
- estimate 100M/500M/1B/2B memory
- resume GPU run

Each command should include one-line explanation.

---

## 15. Test Coverage Additions

Add tests:

```text
tests/test_release_polish.py
tests/test_cli_help.py
tests/test_config_templates.py
tests/test_error_messages.py
```

Equivalent consolidated files are fine.

Minimum test cases:

1. package import works.
2. version is 0.45.0.
3. CLI version returns 0.45.0.
4. major CLI help commands return success.
5. config templates validate or dry-run.
6. configs/jobs validate or estimate without executing large jobs.
7. invalid config path returns clean error.
8. invalid JSON returns clean error.
9. invalid GPU device required error is clean on CPU-only.
10. README contains v0.45/v1.0-beta positioning or equivalent.
11. required docs files exist.
12. command cookbook exists.
13. release checklist exists.
14. smoke example runner quick mode works.
15. release_check script works.
16. no CUDA required.
17. existing Goal 1–44 tests still pass.

---

# Required Verification

Run full tests:

```bash
python -m pytest -q
```

Run release scripts:

```bash
python scripts/release_check.py
python scripts/run_smoke_examples.py --quick
```

Run representative CLI:

```bash
mopforge version
mopforge --help
mopforge config --help
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu launch-torchrun configs/jobs/multigpu_mop_torchrun_plan.json --dry-run
```

Run representative examples:

```bash
python examples/runtime_detection.py
python examples/gpu_train_tiny_smoke.py
python examples/gpu_memory_estimate.py
python examples/gpu_job_profile_validate.py
python examples/manage_models.py
python examples/manage_datasets.py
python examples/analyze_results.py
python examples/build_paper_report.py
```

CUDA optional:

```bash
mopforge gpu train configs/jobs/tiny_gpu_smoke.json --device cuda --precision bf16
```

Only run CUDA command if CUDA is available.

---

# README Update Requirements

Update README to be cleaner and more navigable.

Must include:

- concise top summary
- installation
- quick CPU demo
- quick GPU beta demo
- capabilities overview
- command overview
- docs index
- current limitations summary
- v0.45.0 status
- v1.0-beta path

Do not remove important warnings:

- local Python verifier is not sandboxed
- CPU remains default
- serious GPU jobs require actual hardware
- not production distributed LLM training
- no FSDP/DeepSpeed/custom kernels yet
- no guaranteed large-scale MoP superiority

---

# Final Agent Response Required

When complete, report:

1. Whether Goal 45 is complete.
2. Final version.
3. Files/docs/scripts added.
4. API cleanup summary.
5. CLI UX cleanup summary.
6. Config template audit result.
7. Example audit result.
8. Release scripts added and results.
9. Docs added and docs index.
10. Tests run and count.
11. Examples run.
12. Manual CLI verification.
13. Remaining limitations.
14. Recommended next step:
    - either first real A100/H100 experiment pack
    - or v1.0-beta tag/release.

---

# Final Reminder

Goal 45 is the polish/hardening pass.

Do not expand the research scope.

Make MoP-Forge look and behave like a serious, honest, reproducible research framework ready for first real GPU experiments.
