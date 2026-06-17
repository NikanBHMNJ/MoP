# Quickstart

Install in editable mode:

```bash
pip install -e .[dev]
```

Check the local environment:

```bash
mopforge version
mopforge doctor
python -m pytest -q
```

Run a CPU-safe path:

```bash
python examples/create_lessons.py
python examples/run_tiny_trainer.py
python examples/run_benchmarks.py
python examples/analyze_results.py
```

Run the GPU beta as validation/planning on any machine:

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
```

On CPU-only machines, GPU train smoke jobs may use CPU fallback when allowed.
That keeps tests portable but does not validate GPU performance.
