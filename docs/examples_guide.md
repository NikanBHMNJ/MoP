# Examples Guide

Examples are small local scripts under `examples/`. They create demo data and
outputs in local directories and should be safe to rerun.

List the curated smoke set:

```bash
python scripts/run_smoke_examples.py --list
```

Run the quick CPU-safe subset:

```bash
python scripts/run_smoke_examples.py --quick
```

Include the tiny GPUTrainer smoke path, which may use CPU fallback:

```bash
python scripts/run_smoke_examples.py --quick --include-gpu-fallback
```

Representative examples:

- `examples/runtime_detection.py`: runtime/device detection.
- `examples/gpu_memory_estimate.py`: memory estimates for job profiles.
- `examples/gpu_job_profile_validate.py`: GPU profile validation.
- `examples/manage_models.py`: model registry demo.
- `examples/manage_datasets.py`: dataset registry demo.
- `examples/analyze_results.py`: local analysis report demo.
- `examples/build_paper_report.py`: Markdown paper report scaffold.

CUDA is optional unless an example explicitly says otherwise.
