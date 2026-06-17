# MoP-Forge v0.45.0 Release Notes

MoP-Forge v0.45.0 is a v1.0-beta release candidate polish pass. It keeps the
v0.44.0 single-GPU research beta intact and makes the project easier to import,
diagnose, document, validate, and smoke-test.

## New

- Curated API policy: `mopforge.public_api`.
- CLI diagnostics: `mopforge doctor`.
- Release checks: `python scripts/release_check.py`.
- Example smoke runner: `python scripts/run_smoke_examples.py --quick`.
- Expanded docs index, quickstart, architecture, config template guide,
  examples guide, API overview, command cookbook, known limitations, and
  research positioning.

## Stable

- Knowledge lesson store and indexed local metadata.
- CPU-safe tiny trainer, SFT, continued-pretraining, experiments, benchmarks,
  analysis, reports, datasets, models, artifacts, and checkpoint/resume.
- Runtime device/precision planning.

## Experimental

- `GPUTrainer` and `mopforge.gpu` remain serious single-GPU beta APIs.
- Fast adapters and generated parameters are experimental.
- Torchrun command generation is dry-run foundation only.

## Quick Demo

```bash
pip install -e .[dev]
mopforge doctor
python scripts/release_check.py
python scripts/run_smoke_examples.py --quick
```

## First GPU Experiment Sequence

1. `mopforge runtime detect`
2. `mopforge gpu train configs/jobs/tiny_gpu_smoke.json`
3. `mopforge gpu train configs/jobs/100m_dense_a100_smoke.json`
4. `mopforge gpu train configs/jobs/100m_mop_a100_smoke.json`
5. `mopforge gpu benchmark <run_id>`
6. estimate and validate `configs/jobs/500m_dense_vs_mop_h100.json`
7. validate `configs/jobs/1b_mop_h100_bf16.json`
8. validate 2B only after 100M/500M are stable

## Limitations

No production FSDP/DeepSpeed, no custom CUDA kernels, no hardened multi-GPU
training, no cloud launcher, no external dataset downloads, no guaranteed
large-scale MoP superiority, and no guaranteed 2B/7B execution on all hardware.
