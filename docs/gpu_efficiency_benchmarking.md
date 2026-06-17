# GPU Efficiency Benchmarking

MoP-Forge can run dense and MoP smoke jobs on a CUDA GPU, but GPU-compatible
does not automatically mean GPU-efficient. Goal 46 adds the metrics and sparse
training modes needed to test whether a MoP run uses fewer trainable or active
parameters than a dense baseline while keeping loss, throughput, and memory in
view.

## GPU-Compatible vs GPU-Efficient

GPU-compatible means the trainer can execute on CUDA/BF16 and write results.
GPU-efficient means the run demonstrates a useful tradeoff, such as lower
trainable parameters, lower active parameter estimates, acceptable throughput,
and comparable loss. A full MoP run with `trainable_param_ratio=1.0` is a
compatibility baseline, not an efficiency test.

## Efficiency Metrics

New GPU runs write nested metrics under:

```json
{
  "metrics": {
    "efficiency": {
      "tokens_per_sec": 0.0,
      "samples_per_sec": 0.0,
      "peak_reserved_gb": null,
      "trainable_param_ratio": 0.0
    }
  }
}
```

Key fields:

- `trainable_params` and `trainable_param_ratio`: how many parameters the
  optimizer updates.
- `active_param_estimate` and `active_param_ratio`: approximate active MoP
  parameters for the observed routing path.
- `tokens_per_sec` and `samples_per_sec`: observed throughput for the run.
- `peak_allocated_gb`, `peak_reserved_gb`, `final_allocated_gb`,
  `final_reserved_gb`: CUDA memory stats when available.
- `checkpoint_size_mb`: size of the latest local checkpoint.
- `active_module_density`, `active_adapter_density`, and
  `generated_condition_density`: routing/fast-parameter density metadata.

On CPU-only runs, CUDA memory fields are `null`.

## Sparse MoP Training Modes

Use these `trainable_policy_mode` values:

- `all`: full training; compatibility baseline.
- `adapters_only`: train fast adapters and router if present; freeze core and
  module blocks.
- `modules_only`: train module-specific blocks; freeze shared core.
- `core_frozen`: train module blocks and adapters; freeze shared core.
- `router_adapters_only`: train router and adapters where present; freeze core
  and module blocks.

Parameter group summaries in `metrics.json` show each group’s total, trainable,
and frozen counts.

## Colab Efficiency Configs

Validate and estimate all profiles before running:

```bash
mopforge gpu validate configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu estimate configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_full_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_adapters_only_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_core_frozen_colab_efficiency.json
mopforge gpu validate configs/jobs/100m_mop_router_adapters_colab_efficiency.json
```

Recommended first comparison:

```bash
mopforge gpu train configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_adapters_only_colab_efficiency.json
mopforge gpu compare-runs <dense_run_id> <adapter_mop_run_id> \
  --output outputs/gpu_efficiency_comparison.json
```

The CLI also writes `outputs/gpu_efficiency_comparison.csv`.

The standalone helper supports more explicit paths:

```bash
python scripts/compare_gpu_runs.py \
  --runs <dense_run_id> <adapter_mop_run_id> \
  --gpu-runs-dir gpu_runs \
  --output-json outputs/gpu_efficiency_comparison.json \
  --output-csv outputs/gpu_efficiency_comparison.csv
```

## Interpretation

Do not judge efficiency from one number. Check:

- Loss: sparse MoP must remain useful, not merely small.
- Trainable parameters: adapter-only/core-frozen MoP should be below dense.
- Active parameters: MoP should activate less than its full total where routing
  is meaningful.
- Throughput: sparse parameter updates can still be slower if routing overhead
  dominates.
- VRAM: CUDA reserved memory often matters more than allocated memory in Colab.

Short 100M smoke tests are debugging evidence, not research conclusions. Treat
them as a gate before longer repeated runs with fixed seeds, equal data, equal
token budgets, and honest reporting.
