# GPU Efficiency Benchmarking

MoP-Forge can run dense and MoP jobs on a single device and write comparable
efficiency artifacts. GPU-compatible means a job runs. GPU-efficient means a
run improves a named efficiency axis while keeping quality acceptable.

This document describes the implemented measurement workflow. It does not claim
that MoP is already better than Dense.

## Evidence vs Capability

The committed evidence is the Goal 46 100M Colab/L4 comparison:

- Dense and MoP Full reached similar eval loss.
- MoP Full was not more efficient.
- MoP Adapter-Only was faster and lighter but had worse eval loss.

The newer warm sparse, activation-cache, routed-FFN, and internal-LoRA features
are implemented to support the next experiment. They need fresh CUDA results
before they can be used as performance claims.

## Metrics Written By GPU Runs

GPU runs write nested metrics under:

```json
{
  "metrics": {
    "efficiency": {
      "tokens_per_sec": 0.0,
      "samples_per_sec": 0.0,
      "peak_reserved_gb": null,
      "trainable_param_ratio": 0.0,
      "active_param_ratio": 0.0
    }
  }
}
```

Important fields:

- `trainable_params` and `trainable_param_ratio`: parameters updated by the
  optimizer.
- `active_param_estimate` and `active_param_ratio`: approximate parameters used
  by the routed path.
- `active_trainable_param_estimate` and `active_trainable_param_ratio`:
  trainable subset of the active path.
- `tokens_per_sec`, `samples_per_sec`, and `step_time_sec`: observed
  throughput.
- `peak_allocated_gb`, `peak_reserved_gb`, `final_allocated_gb`, and
  `final_reserved_gb`: CUDA memory stats when available.
- `checkpoint_size_mb`: size of the latest checkpoint artifact.
- `estimated_active_flop_ratio` and `estimated_backward_flop_ratio`: planning
  estimates for routed or frozen paths.
- `generation_eval`: generated-code exact-match and verifier-pass metrics when
  enabled.

On CPU-only runs, CUDA memory fields are `null`; those runs validate behavior,
not GPU performance.

## Sparse Training Modes

Use `trainable_policy_mode` to define what trains:

- `all`: full training; quality or compatibility baseline.
- `adapters_only`: train fast adapters and optional router/head/norm flags.
- `adapters_norm_head`: train adapters plus final norm and LM head.
- `modules_only`: train module-specific blocks; freeze shared core.
- `core_frozen`: train sparse module paths while freezing the shared core.
- `router_only`: train routing parameters only.
- `router_adapters_only`: train router and adapters where present.

Parameter group summaries in `metrics.json` show total, trainable, and frozen
counts for each group.

## Warm Sparse Workflow

The recommended next experiment starts from a learned full-MoP or Dense
checkpoint instead of training sparse adapters from a random frozen base.

```bash
mopforge gpu prepare-efficiency-data --count-per-category 100 --split-seed 42
mopforge gpu train configs/jobs/100m_dense_extended_efficiency.json
mopforge gpu train configs/jobs/100m_mop_full_extended_efficiency.json
mopforge gpu write-warm-sparse-sweep \
  --base-checkpoint <mop_full_run_id_or_checkpoint> \
  --dataset-ref <dataset_id@version_id> \
  --dataset-split-id <split_id> \
  --output-dir configs/jobs/warm_sparse_sweep
```

The sweep generator creates adapter, adapter+norm/head, core-frozen, cached
tail, and routed low-rank profiles. Generated profiles keep the same token
budget and fixed split metadata.

## Activation Caches

Activation caches are for frozen-prefix sparse-tail training. The cache writer
stores hidden states, attention masks, labels, target modules, source IDs, and
config/checkpoint hashes.

```bash
mopforge gpu cache-activations \
  configs/jobs/100m_mop_warm_adapters_norm_head_64_colab_efficiency.json \
  --checkpoint <mop_full_run_id_or_checkpoint> \
  --output outputs/warm_sparse_cache.pt
```

The cache path refuses unsafe runs where the encoded prefix, module bank, or
routed expert blocks are still trainable. This prevents stale activation caches
from being treated as equivalent training data.

## Routed Expert And Low-Rank Paths

The routed-FFN path separates a shared trunk from routed expert blocks. Dense
or post-core checkpoints can warm-start routed blocks by cloning learned dense
FFN weights into each expert.

The routed low-rank path adds zero-initialized deltas inside:

- attention Q/K/V projections,
- attention output projection,
- FFN up projection,
- FFN down projection.

Because the deltas start at zero, enabling the path should not change the warm
base output before training. This is a quality-recovery path, not a proven GPU
efficiency result yet.

## Comparing And Gating Runs

Compare runs:

```bash
mopforge gpu compare-runs <dense_run_id> <sparse_run_id> \
  --output outputs/gpu_efficiency_comparison.json \
  --output-csv outputs/gpu_efficiency_comparison.csv
```

Gate a sparse claim:

```bash
mopforge gpu gate-efficiency \
  --dense-run <dense_run_id> \
  --sparse-run <sparse_run_id> \
  --output outputs/gpu_efficiency_gate_report.json
```

Do not claim same-quality sparse efficiency unless:

- eval loss is close to Dense,
- generated-code verifier pass rate is close to Dense,
- throughput is not materially worse,
- the run improves a named axis such as VRAM, trainable params, checkpoint
  size, cached-tail training time, or active expert compute.

## Interpretation Rules

Use cautious language:

- "lower trainable parameter ratio" is not the same as "lower active compute."
- "lower checkpoint size" is not the same as "lower VRAM."
- CPU fallback is not GPU performance evidence.
- A single short run is debugging evidence, not a research conclusion.
- Any `3x` to `50x` statement must name the exact axis and cite the report.
