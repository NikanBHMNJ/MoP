# Efficiency Report Template

Use this template for production sparse-efficiency reports. Fill it only with
measured run outputs. Leave cells blank until the corresponding run exists.

## Scope

| Field | Value |
| --- | --- |
| Report ID |  |
| Dataset ref |  |
| Split ID |  |
| Split seed |  |
| Split stratification |  |
| Tokenizer |  |
| Quality format |  |
| Target eval loss |  |
| Optimizer updates |  |
| Microsteps |  |
| Gradient accumulation |  |
| Micro batch size |  |
| Max sequence length |  |
| Precision |  |
| Device and memory tier |  |

## Profiles

| Role | Config | Run ID | Checkpoint |
| --- | --- | --- | --- |
| Dense baseline |  |  |  |
| MoP Full baseline |  |  |  |
| Warm sparse baseline |  |  |  |
| Cached sparse adapter |  |  |  |
| Cached tail-only LoRA |  |  |  |

## Results

| Profile | Train loss | Eval loss | Best eval | Target sec | Tokens/sec | Peak alloc GB | Peak reserved GB | Final reserved GB | Trainable ratio | Checkpoint MB | Exact | Syntax | Verifier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MoP Full |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Warm Sparse |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cached Sparse Adapter |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cached Tail-Only LoRA |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Required Controls

| Control | Required | Observed | Pass |
| --- | --- | --- | --- |
| Same train/eval/test split | yes |  |  |
| Same tokenizer and context length | yes |  |  |
| Same eval cadence | yes |  |  |
| Full held-out eval or justified subset | yes |  |  |
| Best-checkpoint generation | yes |  |  |
| Ground-truth verifier controls | 100% |  |  |
| Category coverage | all declared categories |  |  |
| Target truncation | zero or justified |  |  |
| No quantization unless separately declared | yes |  |  |
| No model/checkpoint artifacts in report | yes |  |  |

## Claim Gate

Run:

```bash
mopforge gpu gate-efficiency \
  --dense-run <dense_run_id> \
  --sparse-run <sparse_run_id> \
  --output outputs/efficiency_gate_report.json
```

Then create a claim card:

```bash
mopforge claim scaffold --report-dir reports/<report-id> \
  --claim-statement "<measured, scoped claim>" \
  --academic-level A2 \
  --product-level P2 \
  --output outputs/<report-id>-claim-card.json

mopforge claim validate outputs/<report-id>-claim-card.json
```

## Interpretation

Use scoped wording:

```text
This report supports only the measured axes on the named dataset split,
hardware target, seed, model size, context length, and training budget.
```

Do not generalize to broader model quality, hardware feasibility, production
service readiness, or customer value without separate evidence.
