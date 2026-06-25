# MoP-Forge Repository Instructions

You are working inside:

```text
C:\Users\GC121\Documents\mop
```

## Repository Position

MoP-Forge is no longer presented as a sequence of historical milestone
experiments.
Keep the repo focused on the current `0.46.0` production-oriented research
framework:

- dense, sparse, cached, and routed training profiles,
- verified code-repair quality measurement,
- A100 and H100 admission probes,
- distributed checkpoint/resume/consolidation,
- standard code evaluation and contamination audit,
- Hugging Face export,
- executable claim governance.

Do not reintroduce milestone-numbered public documentation, historical report
folders, or old version narratives. Current public evidence belongs under:

```text
reports/verified_code_repair_100m_l4/
reports/a100_1b_feasibility_probe/
reports/h100_2b_readiness/
reports/claim_readiness_template/
```

Current notebooks are:

```text
notebooks/colab_l4_verified_code_repair_100m.ipynb
notebooks/colab_a100_1b_feasibility_probe.ipynb
notebooks/colab_h100_2b_readiness.ipynb
```

## Current Measured Claim

The only current measured production-facing claim is narrow:

```text
MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4
report split.
```

It is supported by:

```text
reports/verified_code_repair_100m_l4/
```

That report supports an A2/P2 claim only. It does not support broad
code-generation, Qwen-class, frontier-class, product-beta, managed-service,
1B/2B feasibility, or paper-ready claims.

## Claim Governance

Every public claim must pass the framework validator:

```powershell
mopforge claim scaffold --report-dir reports/verified_code_repair_100m_l4 `
  --claim-statement "MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4 report split." `
  --academic-level A2 --product-level P2 `
  --output reports/verified_code_repair_100m_l4/claim_card.json

mopforge claim validate reports/verified_code_repair_100m_l4/claim_card.json
```

Claim standards:

```text
docs/academic_claim_standard.md
docs/startup_product_claim_standard.md
reports/claim_readiness_template/
```

Use A0-A5 for academic claims and P0-P4 for product claims. Strong comparative
research claims require repeated evidence. Public benchmark/model comparisons
require external benchmark and contamination evidence. Product-beta language
requires reliability, monitoring, recovery, security, and support evidence.

## Production Evidence Rules

Do not claim an efficiency win unless the report names the exact improved axis:

- held-out loss,
- generated-code verifier pass,
- exact match,
- syntax pass,
- tokens/sec,
- time-to-target loss,
- peak allocated VRAM,
- peak reserved VRAM,
- final reserved VRAM,
- trainable parameter ratio,
- checkpoint size,
- hardware target.

Do not claim same-quality sparse efficiency unless the same split, tokenizer,
sequence length, batch policy, eval cadence, target metric, and generation
budget are used across comparable profiles.

Do not claim A100 or H100 feasibility from static estimates. The report must
include measured allocator telemetry, finite/decreasing loss, no OOM, exact
optimizer-update count, checkpoint save/load/resume, and runtime projection.

## Artifact Policy

Keep large and sensitive artifacts out of Git:

```text
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.npy
*.npz
activation caches
optimizer states
token shards
training corpora
tokenizer training outputs
model exports
```

Reports must stay lightweight JSON, CSV, Markdown, and small metadata files.

## Implementation Direction

Prefer extending the existing implementation:

```text
mopforge/gpu/
mopforge/models/
mopforge/eval/
mopforge/posttrain/
mopforge/tokenization/
mopforge/claims/
configs/jobs/
docs/
reports/
tests/
```

Keep implementation changes scoped and testable. Use existing config, report,
and CLI patterns before introducing new abstractions.

## Validation

Before finishing code or repo-structure changes, run:

```powershell
python -m pytest -q
python scripts/release_check.py --quick-examples
git diff --check
```

For claim/report changes, also run:

```powershell
mopforge claim scaffold --report-dir reports/verified_code_repair_100m_l4 `
  --claim-statement "MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4 report split." `
  --academic-level A2 --product-level P2 `
  --output reports/verified_code_repair_100m_l4/claim_card.json `
  --validation-output reports/verified_code_repair_100m_l4/claim_validation.json
```

A deliberately overbroad Qwen/frontier/product-beta claim should fail
validation until external benchmark, repeated-seed, contamination, and
reliability evidence exists.
