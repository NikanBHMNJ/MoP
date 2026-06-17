# Contributing

MoP-Forge is CPU-first and local-first. Contributions should keep the full test
suite runnable on machines without CUDA.

Before opening a change:

```bash
python -m pytest -q
python scripts/release_check.py
python scripts/run_smoke_examples.py --quick
```

Guidelines:

- Do not commit large generated run outputs, datasets, checkpoints, or local
  registries.
- Keep examples tiny, idempotent, and CPU-safe unless they explicitly guard CUDA.
- Gate CUDA execution with runtime detection.
- Prefer existing config, registry, runtime, and artifact helpers over new
  parallel systems.
- Document limitations plainly; do not imply production distributed training.

Optional CUDA checks are welcome when hardware is available, but CPU-only tests
must remain sufficient for normal CI.
