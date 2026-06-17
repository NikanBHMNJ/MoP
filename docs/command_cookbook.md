# Command Cookbook

Create lessons:

```bash
python examples/create_lessons.py
```

Register a dataset:

```bash
mopforge dataset register data/coding_bugfix_lessons.jsonl --name coding_bugfix --kind lessons
```

Register a model:

```bash
mopforge model register configs/examples/model_tiny_mop.json
```

Run CPU trainer:

```bash
mopforge train run configs/examples/tiny_trainer_mop_cpu.json
```

Run SFT:

```bash
mopforge sft run configs/examples/sft_full_cpu.json
```

Run benchmark:

```bash
mopforge benchmark run configs/examples/benchmark_composite.json
```

Run analysis:

```bash
mopforge analyze compare --run-path runs/example/trainer_result.json
```

Build paper report:

```bash
mopforge paper build configs/examples/paper_report_smoke.json
```

Detect runtime:

```bash
mopforge runtime detect
```

Validate GPU config:

```bash
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
```

Run tiny GPU smoke:

```bash
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
```

Estimate 100M/500M/1B/2B memory:

```bash
mopforge gpu estimate configs/jobs/100m_mop_a100_smoke.json
mopforge gpu estimate configs/jobs/500m_dense_vs_mop_h100.json
mopforge gpu estimate configs/jobs/1b_mop_h100_bf16.json
mopforge gpu estimate configs/jobs/2b_mop_a100_plan.json
```

Resume GPU run:

```bash
mopforge gpu resume <checkpoint_or_run_id>
```
