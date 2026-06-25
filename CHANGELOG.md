# Changelog

## 0.46.0

MoP-Forge is currently shipped as a production-oriented research framework for
measured dense, sparse, cached, routed, and verified code-model workflows.

- Published the current verified code-repair efficiency report under
  `reports/verified_code_repair_100m_l4/`.
- Added semantic A100 and H100 readiness report targets under
  `reports/a100_1b_feasibility_probe/` and `reports/h100_2b_readiness/`.
- Added executable claim governance through `mopforge claim scaffold` and
  `mopforge claim validate`.
- Added academic and product claim standards, with explicit supported and
  blocked wording.
- Consolidated public notebooks around current L4/A100/H100 workflows.
- Kept model weights, optimizer state, token shards, activation caches, and
  checkpoints outside Git.

This release supports narrow measured code-repair efficiency claims only. It
does not claim broad code generation, managed-service readiness, 1B/2B
feasibility, or paper-ready comparative results until the corresponding reports
are produced and pass claim validation.
