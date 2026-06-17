# Serious Jobs Checklist

MoP-Forge now includes a serious single-GPU research beta for tiny-to-small MoP
experiments and validated large-job profiles. It is not yet a fully production
distributed LLM training framework.

Run the first real hardware experiments in this exact order:

1. runtime detect
2. tiny GPU smoke
3. 100M dense
4. 100M MoP
5. dense vs MoP benchmark
6. 500M dense vs MoP
7. validate 1B
8. validate 2B only after 100M/500M are stable

## Commands

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/100m_dense_a100_smoke.json
mopforge gpu train configs/jobs/100m_dense_a100_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu train configs/jobs/100m_mop_a100_smoke.json
mopforge gpu benchmark <run_id>
mopforge gpu show <run_id>
mopforge gpu estimate configs/jobs/500m_dense_vs_mop_h100.json
mopforge gpu validate configs/jobs/1b_mop_h100_bf16.json
mopforge gpu validate configs/jobs/2b_mop_a100_plan.json
```

## Expected Signals

`runtime detect` should show `cuda_available=True` and the expected device
count/name on GPU hardware. `gpu validate` should print `validation=valid` and
may print warnings for planning-only profiles. `gpu train` writes a local
`gpu_runs/<run_id>/` directory with `config.json`, `runtime.json`,
`metrics.json`, `state.json`, `gpu_training_result.json`, `memory_estimate.json`,
and checkpoints.

On CPU-only machines, tiny GPU smoke can fall back to CPU when the profile sets
`require_device_available=false`; that is useful for CI but does not prove GPU
performance.

## Troubleshooting

- CUDA unavailable: install a CUDA-capable PyTorch build for the machine, then
  rerun `mopforge runtime detect`.
- Memory estimate exceeds target GPU: lower `max_seq_len`, `micro_batch_size`,
  or increase gradient accumulation before training.
- Missing data: create or register local JSONL lesson/corpus files; MoP-Forge
  does not download datasets.
- Missing checkpoint/run ID: run `mopforge gpu list`, then `mopforge gpu show
  <run_id>` to inspect available local runs.
- Large 1B/2B/7B configs are validation/planning profiles unless the user
  explicitly runs them on suitable hardware and data.
