# Changelog

## 0.45.0

Goal 45 hardens the v0.44.0 serious GPU beta into a v1.0-beta release
candidate.

- Added a curated public API policy in `mopforge.public_api`.
- Added `mopforge doctor` for local environment diagnostics.
- Improved CLI debug/error formatting and GPU command wording.
- Hardened config loading errors, GPU ref validation, and resume messages.
- Added release docs, command cookbook, installation guide, config template
  guide, examples guide, known limitations, and research positioning.
- Added `scripts/release_check.py` and `scripts/run_smoke_examples.py`.
- Added release metadata, repository hygiene docs, and v0.45.0 tests.

Known limitations remain intentional: no production FSDP/DeepSpeed, no custom
CUDA kernels, no production distributed training, FP8 planning-only, and large
2B/7B profiles are planning artifacts unless explicitly run on suitable
hardware and data.

## 0.44.0

Goal 37-44 introduced the serious single-GPU research beta:

- `mopforge.gpu` package with `GPUTrainingConfig`, `GPUTrainer`, run registry,
  data loaders, checkpoint/resume, memory estimation, and GPU job validation.
- AMP/GradScaler wrapper, gradient accumulation, activation-checkpoint metadata,
  and efficient-attention planning metadata.
- A100/H100 job profiles and torchrun dry-run launcher foundation.
- PyTorch-level MoP routing/Fast-Parameter metadata for GPU experiments.
- `mopforge gpu` CLI for validate, estimate, train, resume, benchmark, list,
  show, and launch-torchrun dry-run.

Upgrade note: CPU remains the default. CUDA-specific execution is optional and
guarded by runtime/device validation.
