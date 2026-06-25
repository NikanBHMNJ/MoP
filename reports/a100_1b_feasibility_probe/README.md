# A100 1B Feasibility Probe

This directory is the lightweight report target for 1B A100 admission probes.
No measured A100 probe has been committed yet.

Run:

```text
notebooks/colab_a100_1b_feasibility_probe.ipynb
```

The notebook detects A100 40 GB versus 80 GB, selects the matching profile, and
exports lightweight report artifacts. Do not add checkpoints, activation
caches, optimizer state, token shards, corpora, or model weights.

## Required Gates

- no OOM,
- finite decreasing loss,
- exact optimizer-update count,
- phase-level allocated/reserved VRAM telemetry,
- successful model-only checkpoint save/load/resume,
- peak reserved VRAM within the hardware-specific limit,
- 500-update and 2,000-update runtime projections.

## Implemented Profiles

- `configs/jobs/1b_dense_a100_40gb_probe.json`
- `configs/jobs/1b_dense_a100_80gb_probe.json`
- `configs/jobs/1b_mop_full_a100_40gb_probe.json`
- `configs/jobs/1b_mop_full_a100_80gb_probe.json`
- `configs/jobs/1b_cached_adapter_128_a100_40gb_probe.json`
- `configs/jobs/1b_cached_adapter_128_a100_80gb_probe.json`

This folder is an admission target, not a measured feasibility claim.
