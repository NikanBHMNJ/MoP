# Claim Readiness Template

Copy this folder when preparing a public academic, investor, product, or
release claim. Fill the claim card before updating the README, paper draft,
website copy, model card, or release notes.

The framework can generate and validate this structure:

```bash
mopforge claim scaffold --report-dir reports/<report-id> \
  --claim-statement "<measured claim sentence>" \
  --academic-level A2 \
  --product-level P2 \
  --output outputs/<claim-id>.json

mopforge claim validate outputs/<claim-id>.json
```

## Claim Card

| Field | Value |
| --- | --- |
| Claim ID | `<release-or-report-claim-id>` |
| Claim statement | `<one sentence>` |
| Claim type | `academic`, `product`, `release`, or `benchmark` |
| Academic level | `A0` to `A5` from `docs/academic_claim_standard.md` |
| Product level | `P0` to `P4` from `docs/startup_product_claim_standard.md` |
| Evidence status | `implemented`, `measured`, `repeated`, `external`, or `blocked` |
| Repo commit | `<git commit>` |
| Version | `<package version>` |
| Report directory | `<reports/...>` |
| Dataset and split | `<dataset ref, split ID, seed, hashes>` |
| Hardware | `<device, memory tier, driver, CUDA, PyTorch>` |
| Baselines | `<dense/full-mop/external baselines>` |
| Primary metric | `<metric that decides pass/fail>` |
| Secondary metrics | `<loss, quality, speed, memory, artifact size>` |
| Allowed wording | `<exact public sentence>` |
| Blocked wording | `<phrases that are not supported>` |
| Decision | `publish`, `publish-narrowly`, `rerun`, or `block` |

## Required Attachments

- `comparison.json` or equivalent metric artifact.
- `comparison.csv` when multiple profiles are compared.
- `run_manifest.json` or per-run manifests.
- `experiment_settings.json` with data, tokenizer, split, context, batch,
  optimizer, precision, and eval settings.
- `acceptance_gates.json` when a pass/fail decision is made.
- Generated samples and verifier output for code-quality claims.
- Contamination audit for external benchmark claims.
- Artifact audit showing no model weights, optimizer states, caches, token
  shards, or checkpoints are committed.

## Decision Rules

- Use `publish` only when the claim level matches the evidence.
- Use `publish-narrowly` when the result is valid but scoped to a dataset,
  hardware target, seed, or task.
- Use `rerun` when the report is incomplete, underseeded, or missing a required
  baseline.
- Use `block` when any control fails or public wording would overclaim.

## Lightweight Artifact Policy

This folder is for claim metadata only. Do not add:

```text
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.npz
*.npy
```
