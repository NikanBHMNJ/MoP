# Goal 46 GPU Efficiency Report

This folder contains the lightweight report artifacts for the Goal 46 100M GPU
efficiency comparison. It intentionally stores JSON and CSV evidence only. It
does not include model checkpoints or large tensor files.

## Purpose

The experiment tested whether MoP-Forge can measure GPU efficiency for dense
and Mixture-of-Parameters training modes, and whether sparse MoP training can
reduce trainable parameters, VRAM use, checkpoint size, or training cost.

This is an evidence artifact, not a claim that MoP is better than dense.

## Hardware And Runtime

- Runtime: Google Colab CUDA runtime
- Device: `cuda:0`
- Precision: `bf16`
- PyTorch: `2.11.0+cu128`
- TF32: enabled
- AMP: enabled

## Run IDs

- Dense: `20260617T094127Z-100m-dense-colab-efficiency-a4e4bd2e`
- MoP Full: `20260617T094141Z-100m-mop-full-colab-efficiency-539270ba`
- MoP Adapter-Only: `20260617T094157Z-100m-mop-adapters-only-colab-efficiency-47bab16e`

## Result Table

| Model            | Train loss | Eval loss | Tokens/sec | Peak reserved VRAM |       Trainable ratio | Active ratio | Checkpoint size | Device |
| ---------------- | ---------: | --------: | ---------: | -----------------: | --------------------: | -----------: | --------------: | ------ |
| Dense            |     3.0467 |    3.1705 | 11286.5449 |          1.9844 GB |                   1.0 |          1.0 |     987.1423 MB | cuda:0 |
| MoP Full         |     3.0377 |    3.1691 | 10402.1507 |          2.1367 GB |                   1.0 |          1.0 |    1078.0514 MB | cuda:0 |
| MoP Adapter-Only |     5.1322 |    5.1653 | 26812.3098 |          0.4961 GB | 0.0008424878500822691 |          1.0 |     365.8617 MB | cuda:0 |

## Interpretation

MoP Full matched dense quality but was not more efficient.

MoP Adapter-Only was much faster, used much less VRAM, had a much smaller
checkpoint, and trained far fewer parameters, but its eval loss was worse.

This is evidence that MoP-Forge can measure GPU efficiency and run sparse MoP
modes, not proof that MoP is better than dense.

Derived comparisons:

- MoP Adapter-Only was about 2.38x faster than dense by tokens/sec.
- MoP Adapter-Only used about 75% less peak reserved VRAM than dense.
- MoP Adapter-Only used about 99.916% fewer trainable parameters than dense.
- MoP Adapter-Only checkpoint was about 63% smaller than dense.

## Artifact Layout

```text
reports/goal46_gpu_efficiency/
  100m_efficiency_comparison.csv
  100m_efficiency_comparison.json
  goal46_summary.json
  runs/
    dense/
    mop_full/
    mop_adapters_only/
```

Each run folder includes:

```text
config.json
gpu_training_result.json
memory_estimate.json
metrics.json
runtime.json
state.json
```

## Reproduction Commands

```bash
pip install -e .[dev]
mopforge doctor
mopforge runtime detect

mopforge gpu train configs/jobs/100m_dense_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_full_colab_efficiency.json
mopforge gpu train configs/jobs/100m_mop_adapters_only_colab_efficiency.json

mopforge gpu compare-runs \
  20260617T094127Z-100m-dense-colab-efficiency-a4e4bd2e \
  20260617T094141Z-100m-mop-full-colab-efficiency-539270ba \
  20260617T094157Z-100m-mop-adapters-only-colab-efficiency-47bab16e \
  --output outputs/100m_efficiency_comparison.json
```

The standalone report helper can also be used:

```bash
python scripts/compare_gpu_runs.py \
  --runs \
  20260617T094127Z-100m-dense-colab-efficiency-a4e4bd2e \
  20260617T094141Z-100m-mop-full-colab-efficiency-539270ba \
  20260617T094157Z-100m-mop-adapters-only-colab-efficiency-47bab16e \
  --gpu-runs-dir gpu_runs \
  --output-json outputs/100m_efficiency_comparison.json \
  --output-csv outputs/100m_efficiency_comparison.csv
```

## Limitations

- This was a short 100M Colab/L4-style efficiency run, not a research
  conclusion.
- MoP Adapter-Only improved throughput and memory but degraded eval loss.
- MoP Full matched dense loss but trained all parameters and used more VRAM.
- Active parameter ratio is still `1.0` for these runs, so this does not yet
  demonstrate sparse active-parameter execution.
- No checkpoints are included in this report folder.
- No DeepSpeed, FSDP, custom CUDA kernels, or multi-GPU claims are involved.

## Next Experiment Recommendation

Run longer dense versus MoP Adapter-Only and core-frozen comparisons with equal
data, fixed seeds, repeated trials, and a configuration that reduces active
parameter ratio below dense while tracking eval loss, tokens/sec, VRAM, and
checkpoint size.
