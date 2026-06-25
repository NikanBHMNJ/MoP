# Startup And Product Claim Standard

MoP-Forge can be positioned as a serious product direction only when claims are
tied to user value, reproducible workflows, operating cost, reliability, and clear
scope. This document separates research evidence from startup/product wording.

## Product Claim Levels

| Level | Name | Meaning | Minimum Evidence |
| --- | --- | --- | --- |
| P0 | Research prototype | The repo demonstrates the technical direction. | Local install, tests, docs, example configs, and report artifacts. |
| P1 | Reproducible workflow | A technical user can rerun a focused workflow. | Notebook or script, fixed dataset, downloadable report, and pass/fail gates. |
| P2 | Pilot-ready workflow | A user can test a narrow job on their hardware. | Hardware admission report, restart/resume, cost metrics, failure modes, and artifact policy. |
| P3 | Product beta | Repeated users can rely on it for a scoped workflow. | Stable interface, monitoring, support docs, security notes, benchmark dashboard, and upgrade path. |
| P4 | Customer-proven product | Real users get repeatable value. | Customer/pilot results, uptime/recovery evidence, cost savings, and support process. |

MoP-Forge is currently closest to P1 for narrow reproducible reports and P2 for
planned A100/H100 admission workflows. It is not yet a P3 managed product.

Use `mopforge claim validate` before product wording is published. P2 and
higher claims must pass hardware, metric, limitation, and artifact gates in a
claim card generated from a real report folder.

## Product Metrics

Product claims should use user-facing metrics, not only research metrics:

- cost per verified repair or accepted completion,
- time to first successful run,
- setup steps and required hardware class,
- pass@1, verifier pass, syntax pass, exact match, and failure categories,
- tokens/sec, samples/sec, time-to-target loss, and time-to-target quality,
- peak reserved VRAM, peak allocated VRAM, host RAM, and disk footprint,
- checkpoint size, trainable-only artifact size, export size, and resume time,
- failure/retry rate, OOM behavior, checkpoint recovery, and report integrity,
- deployment latency and generation throughput after export,
- security boundary for code execution and verifier trust assumptions.

## Current Product Positioning

Use this positioning until broader evidence exists:

```text
MoP-Forge is an evidence-first training and evaluation framework for measuring
cost-efficient dense, sparse, cached, and routed code-model experiments. Its
strongest current result is narrow verified code-repair efficiency at 100M
scale; larger A100/H100 profiles are admission workflows pending hardware
evidence.
```

This is a credible startup story because it names the customer pain: reducing
the cost of verified specialist training while preserving measured quality.

## Product Claims Allowed Today

- "Reproducible local and notebook workflows for narrow code-repair efficiency
  reports."
- "Trainable-only sparse checkpoints and cached sparse-tail training reduce
  the measured artifact and training-phase memory footprint in completed
  reports."
- "A100 and H100 admission notebooks are designed to stop before expensive
  training if memory, resume, and throughput gates fail."
- "The framework records report artifacts needed to audit quality, efficiency,
  and failure modes."

## Product Claims Not Yet Allowed

- "Drop-in replacement for Qwen, CodeLlama, or frontier code models."
- "Production-grade managed training platform."
- "Guaranteed 1B or 2B training on arbitrary A100/H100 environments."
- "General code-generation quality" from the narrow verified repair benchmark.
- "Customer-proven cost reduction" before external pilots exist.

## Pilot Requirements

Before calling the project pilot-ready for a user, produce a claim card with:

1. A named user workflow, such as verified repair fine-tuning or cached sparse
   admission probing.
2. A supported hardware target and measured runtime/cost estimate.
3. A passing report from a clean clone.
4. A failure-mode section covering OOM, bad data, verifier failures, resume
   failures, and missing artifact paths.
5. A security note for code execution and dataset trust.
6. A supportable "what it does" and "what it does not do" statement.

## Startup Narrative

The strongest near-term product narrative is:

```text
Small and medium teams can use MoP-Forge to measure whether sparse cached
specialist training reduces verified-code training cost before committing to a
larger model run.
```

That narrative should remain tied to measured reports. When the H100 2B path
passes standard code benchmarks, the narrative can expand from "framework and
admission workflow" to "released specialist model pipeline."
