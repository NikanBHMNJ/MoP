# GPU Job Profiles

Tracked profiles live under `configs/jobs/`.

- `tiny_gpu_smoke.json`: executable CPU fallback and CUDA smoke profile.
- `100m_dense_a100_smoke.json`: small dense A100 profile.
- `100m_mop_a100_smoke.json`: small MoP A100 profile.
- `500m_dense_vs_mop_h100.json`: validation/planning profile.
- `1b_mop_h100_bf16.json`: validation/planning profile.
- `2b_mop_a100_plan.json`: validation/planning profile.
- `7b_mop_h100_plan.json`: validation/planning profile.
- `multigpu_mop_torchrun_plan.json`: torchrun dry-run foundation profile.

Estimate before running:

```bash
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu validate configs/jobs/2b_mop_a100_plan.json
```

The memory estimator is approximate. It accounts for weights, gradients,
AdamW-style optimizer state, and rough activations. It does not prove a 2B or
7B profile will fit or converge.
