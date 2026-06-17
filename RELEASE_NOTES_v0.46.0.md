# MoP-Forge v0.46.0 Release Notes

MoP-Forge v0.46.0 turns the single-GPU beta into a more useful efficiency
benchmarking workflow. It does not claim MoP superiority; it adds the metrics,
configs, docs, and comparison tools needed to test dense versus sparse MoP runs
more honestly.

## New

- GPU run efficiency metrics under `metrics.efficiency`.
- CUDA peak/current allocated and reserved memory tracking when CUDA is active.
- Sparse trainable policies: `adapters_only`, `modules_only`, `core_frozen`,
  and `router_adapters_only`.
- Colab-safe 100M dense and MoP efficiency config templates.
- `mopforge gpu compare-runs` for JSON/CSV/table comparisons.
- `scripts/compare_gpu_runs.py` for standalone benchmark reports.
- GPU efficiency benchmarking docs and Colab corpus helper docs.

## Stable

- CPU-first smoke tests and release checks remain the required baseline.
- Existing GPU training, resume, validation, and estimate commands remain
  backward compatible with older result JSON structures.
- Existing v0.45.0 public API policy remains in place.

## Experimental

- GPUTrainer and MoP routing remain single-device research beta features.
- Sparse MoP training modes are meant for measurement and iteration, not
  research conclusions from short smoke runs.
- Fast adapters and generated parameters remain experimental.

## Quick Demo

```bash
pip install -e .[dev]
mopforge gpu validate configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_adapters_only_colab_efficiency.json
mopforge gpu compare-runs <dense_run_id> <adapter_mop_run_id> \
  --output outputs/gpu_efficiency_comparison.json
```

## Limitations

No production FSDP/DeepSpeed, no custom CUDA kernels, no hardened multi-GPU
training, no cloud launcher, no external dataset downloads by default, no
guaranteed large-scale MoP superiority, and no research claim from short smoke
benchmarks.
