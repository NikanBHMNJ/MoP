# Release Checklist

Run these checks before tagging or sharing a v0.45.0 release candidate:

```bash
python -m pytest -q
python scripts/release_check.py
python scripts/release_check.py --quick-examples
python scripts/run_smoke_examples.py --quick
```

Manual CLI spot checks:

```bash
mopforge version
mopforge --help
mopforge doctor
mopforge config --help
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu launch-torchrun configs/jobs/multigpu_mop_torchrun_plan.json --dry-run
```

Do not tag if CPU tests fail, docs are missing, or GPU planning commands crash
on CPU-only machines.
