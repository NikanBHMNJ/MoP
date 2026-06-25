# Academic Claim Standard

MoP-Forge should make research claims only at the evidence level the repository
actually supports. This document defines the claim ladder used by README files,
reports, papers, abstracts, and benchmark summaries.

## Claim Levels

| Level | Name | Meaning | Minimum Evidence |
| --- | --- | --- | --- |
| A0 | Implemented mechanism | The code path exists and passes local tests. | Tests, config examples, and no performance wording. |
| A1 | Single-run smoke evidence | One run proves the workflow can execute. | Run manifest, hardware, config, metrics, and artifact audit. |
| A2 | Controlled comparison | One fixed split compares baselines and MoP profiles. | Same data, tokenizer, sequence length, batch policy, eval cadence, and target metric. |
| A3 | Repeated comparison | The result survives reruns. | At least three seeds or repeated runs, variance/error bars, and negative-result notes. |
| A4 | External benchmark evidence | The result holds on recognized benchmarks or trusted task suites. | External baselines, contamination audit, ablations, standard metrics, and reproducible commands. |
| A5 | Paper-ready claim | The result can support a paper headline. | A4 evidence plus limitations, artifact release plan, independent reproduction path, and statistical analysis. |

Use the lowest level that matches the evidence. A higher-level implementation
does not raise the claim level until the measurements exist.

## Framework Commands

Claim readiness is enforced by the MoP-Forge CLI and Python API:

```bash
mopforge claim scaffold \
  --report-dir reports/verified_code_repair_100m_l4 \
  --claim-statement "MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4 report split." \
  --academic-level A2 \
  --product-level P2 \
  --output reports/verified_code_repair_100m_l4/claim_card.json

mopforge claim validate reports/verified_code_repair_100m_l4/claim_card.json
```

The same logic is available as `mopforge.claims.validate_claim_card` and
`mopforge.public_api.validate_claim_card`.

## Required Evidence Bundle

Every academic claim must point to a lightweight report folder and include:

- repository commit and package version,
- exact config paths and command lines,
- dataset name, split ID, split seed, data hashes, tokenizer source hashes, and
  contamination audit when applicable,
- model profile, parameter count, trainable parameter ratio, active parameter
  ratio when routed execution is used, and checkpoint policy,
- hardware name, memory tier, driver, CUDA, PyTorch, precision, and distributed
  topology,
- optimizer-step budget, token budget, microsteps, gradient accumulation,
  eval cadence, save cadence, and early-stop policy,
- train loss, eval loss, best eval loss, time-to-target loss, tokens-to-target
  loss, tokens/sec, samples/sec, peak allocated VRAM, peak reserved VRAM, final
  reserved VRAM, host memory, and checkpoint size,
- generated-code quality when code tasks are used: syntax pass, exact match,
  verifier/task pass, pass@1, per-category failures, samples, and trusted
  ground-truth controls,
- baseline runs that use the same split, tokenizer, context length, batch
  policy, eval cadence, and target metric,
- artifact audit proving checkpoints, optimizer states, token shards, caches,
  and model weights are excluded from Git.

## Allowed Current Wording

The current repository can support wording like:

- "MoP-Forge implements an evidence-first framework for Dense, MoP Full, warm
  sparse, cached sparse, LoRA, routed-expert, and distributed training
  experiments."
- "The verified code-repair report supports a narrow 100M L4 efficiency claim
  on the measured fixed split and hardware."
- "The A100 and H100 report targets provide admission profiles and schemas, but
  hardware feasibility is not claimed until measured reports are generated."

## Blocked Wording

Do not use these claims until the matching evidence level exists:

- "MoP-Forge beats dense training" without same-split repeated baselines.
- "Qwen-class", "frontier-class", or "usable 2B model" without standard
  benchmark pass@1, contamination evidence, generated samples, deployment
  profile, and at least a second seed.
- "3x to 50x lower GPU memory" unless measured allocated and reserved VRAM
  show that improvement for the named training phase and hardware.
- "Same quality" unless held-out loss and task-quality metrics remain within
  the predeclared acceptance band.
- "Production training service" unless cloud launch, monitoring, recovery,
  security, and support requirements have been implemented and tested.

## Paper-Ready Checklist

Before writing an academic abstract or paper claim, require:

1. A completed claim card using `reports/claim_readiness_template/`.
2. A0 through A2 evidence for every implementation claim in the paper.
3. A3 or higher evidence for comparative efficiency or quality claims.
4. A4 or higher evidence for claims against external models or public
   benchmarks.
5. Ablations for cache use, trainable policy, teacher distillation, replay, and
   routed experts when those features are part of the conclusion.
6. Limitations that name failed runs, unsupported hardware, local verifier
   trust assumptions, benchmark scope, and hidden artifact exclusions.

## Interpretation Rule

An implementation can be impressive without being a measured research result.
An experiment can be useful without being paper-ready. Always label the claim
by its evidence level and keep the strongest sentence no stronger than the
weakest required metric.
