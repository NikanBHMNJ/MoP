# MoP-Forge Documentation

These docs cover MoP-Forge `0.46.0`, the current production-oriented research
framework for dense, sparse, cached, and routed code-model training workflows.
Historical milestone documentation has been removed from the primary surface;
current evidence and readiness artifacts are listed in
[reports/](../reports/).

## Start Here

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [GPU quickstart](gpu_quickstart.md)
- [Architecture](architecture.md)
- [Public API overview](api_overview.md)
- [Command cookbook](command_cookbook.md)

## Production Training And Evaluation

- [Production 2B readiness](production_2b_readiness.md)
- [GPU job profiles](gpu_job_profiles.md)
- [GPU efficiency benchmarking](gpu_efficiency_benchmarking.md)
- [Efficiency report template](efficiency_report_template.md)
- [GPU runtime limitations](gpu_runtime_limitations.md)
- [Serious jobs checklist](serious_jobs_checklist.md)
- [Colab training notes](colab_training.md)

## Current Notebooks

- [L4 verified code repair 100M](../notebooks/colab_l4_verified_code_repair_100m.ipynb)
- [A100 1B feasibility probe](../notebooks/colab_a100_1b_feasibility_probe.ipynb)
- [H100 2B readiness](../notebooks/colab_h100_2b_readiness.ipynb)

## Reports

- [Reports index](../reports/README.md)
- [Verified code repair 100M L4](../reports/verified_code_repair_100m_l4/README.md)
- [A100 1B feasibility probe](../reports/a100_1b_feasibility_probe/README.md)
- [H100 2B readiness](../reports/h100_2b_readiness/README.md)
- [Claim readiness template](../reports/claim_readiness_template/README.md)

## Claim Governance

- [Academic claim standard](academic_claim_standard.md)
- [Startup and product claim standard](startup_product_claim_standard.md)
- [Research positioning](research_positioning.md)
- [Known limitations](known_limitations.md)

Claim readiness is executable:

```bash
mopforge claim scaffold --report-dir reports/verified_code_repair_100m_l4 \
  --claim-statement "MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4 report split." \
  --academic-level A2 --product-level P2 \
  --output reports/verified_code_repair_100m_l4/claim_card.json

mopforge claim validate reports/verified_code_repair_100m_l4/claim_card.json
```

## Implementation And Release Context

- [Config templates](config_templates.md)
- [Examples guide](examples_guide.md)
- [Release checklist](release_checklist.md)

## Current Evidence Summary

The current measured report is
`reports/verified_code_repair_100m_l4/`. It compares Dense, MoP Full, Warm
Adapter/Norm/Head 128, Cached Adapter/Norm/Head 128, and Cached Tail-Only LoRA
Rank 8 on the same fixed 10,000-lesson verified code-repair split.

The narrow measured result:

- Cached Adapter/Norm/Head 128 reached `88.0%` verifier and exact match versus
  Dense at `82.4%`.
- Cached Adapter/Norm/Head 128 measured `8.35x` Dense throughput and `31.70x`
  lower peak reserved VRAM.
- Cached Tail-Only LoRA Rank 8 reached the same `88.0%` verifier and exact
  match with `6.70x` Dense throughput and `22.83x` lower peak reserved VRAM.
- All claims remain scoped to the report dataset, split, seed, model size,
  hardware, and training budget.

The A100 and H100 report folders are admission targets, not measured hardware
claims yet.
