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
structured data, tiny models, local checkpoints, experiments, benchmarks,
analysis, reports, and single-device GPU beta jobs can be configured and tested.

## What It Does Not Prove Yet

It does not prove large-scale MoP quality, production training stability, or
distributed efficiency.

## First Useful GPU Evidence

Successful first evidence would be stable tiny GPU smoke, comparable 100M dense
and MoP runs, repeatable dense-vs-MoP benchmarks, and a 500M H100 comparison
with enough run metadata to inspect losses, routing density, trainable
parameter ratios, and memory behavior.
