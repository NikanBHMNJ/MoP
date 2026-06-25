# Notebook Training

MoP-Forge keeps notebooks as operational runbooks for measured report
generation. They are not the source of truth for claims; reports and claim cards
are.

## Current Notebooks

- `notebooks/colab_l4_verified_code_repair_100m.ipynb`
- `notebooks/colab_a100_1b_feasibility_probe.ipynb`
- `notebooks/colab_h100_2b_readiness.ipynb`

Each notebook writes lightweight report artifacts under `reports/` and excludes
checkpoints, optimizer state, activation caches, token shards, corpora, and
model weights.

## L4 Verified Code Repair

Run the L4 notebook when validating the current 100M verified repair workflow.
It prepares a fixed, stratified code-repair split, trains comparable Dense/MoP
profiles, evaluates generated repairs from best checkpoints, writes
`comparison.json`, `acceptance_gates.json`, and generated-code artifacts, then
packages the report folder.

Expected output:

```text
reports/verified_code_repair_100m_l4/
```

## A100 And H100 Admission

The A100 and H100 notebooks are staged admission workflows. They should stop
before expensive training when memory, OOM, loss, checkpoint, or resume gates
fail.

Expected output targets:

```text
reports/a100_1b_feasibility_probe/
reports/h100_2b_readiness/
```

## Claim Validation

After downloading or committing a lightweight report, run:

```bash
mopforge claim scaffold --report-dir reports/<report-id> \
  --claim-statement "<measured, scoped claim>" \
  --academic-level A2 \
  --product-level P2 \
  --output outputs/<report-id>-claim-card.json

mopforge claim validate outputs/<report-id>-claim-card.json
```
