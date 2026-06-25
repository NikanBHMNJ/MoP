# Research Positioning

MoP-Forge is a local-first research framework for testing
Mixture-of-Parameters training ideas. It provides reproducible
data/model/run/benchmark/report infrastructure and a serious single-GPU
research beta, but it has not yet demonstrated large-scale MoP superiority.

## What Problem It Explores

MoP-Forge explores whether structured lesson metadata, module-specific
parameters, routers, fast adapters, generated parameters, feedback, and
curriculum signals can make parameter use more targeted and inspectable than a
single dense parameter path.

## MoP vs MoE

Mixture-of-Experts usually routes tokens or examples through expert modules.
MoP-Forge experiments with a broader parameter-family view: stable core
weights, named module parameters, router parameters, fast adapters, generated
adapter tensors, and feedback/curriculum control. The implementation is
research plumbing, not a claim that MoP is better than MoE today.

## What It Proves Today

It proves that the local research loop can be made reproducible and CPU-safe:
structured data, local checkpoints, experiments, benchmarks, analysis, reports,
single-device GPU jobs, and torchrun DDP/FSDP jobs can be configured and
tested.

## Claim Discipline

MoP-Forge separates implementation, measurement, academic claims, and product
claims. Use [academic_claim_standard.md](academic_claim_standard.md) for A0 to
A5 research wording and [startup_product_claim_standard.md](startup_product_claim_standard.md)
for P0 to P4 startup/product wording. Public claims should also include a
claim card based on [the report template](../reports/claim_readiness_template/README.md).
The same standard is implemented by `mopforge claim scaffold`,
`mopforge claim validate`, and the `mopforge.claims` Python API.

The strongest current product sentence is narrow:

```text
MoP-Forge is an evidence-first training and evaluation framework for measuring
cost-efficient dense, sparse, cached, and routed code-model experiments.
```

Do not call it a Qwen-class model, a frontier model, or a production training
service until the matching reports, benchmarks, reliability evidence, and user
workflow proof exist.

## What It Does Not Prove Yet

It does not prove large-scale MoP quality, production training stability, or
distributed efficiency.

## First Useful GPU Evidence

Successful first evidence would be stable tiny GPU smoke, comparable 100M dense
and MoP runs, repeatable dense-vs-MoP benchmarks, and a 500M H100 comparison
with enough run metadata to inspect losses, routing density, trainable
parameter ratios, and memory behavior.
