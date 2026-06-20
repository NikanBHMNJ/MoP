# MoP-Forge Documentation

These docs cover MoP-Forge `0.46.0`, a local-first research framework for
Mixture-of-Parameters experiments and single-GPU efficiency studies.

The docs are organized around one principle: implementation details, experiment
evidence, and research claims should stay separate. A feature can be implemented
and tested before it has proven a GPU efficiency result.

## Start Here

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Architecture](architecture.md)
- [Public API overview](api_overview.md)
- [Command cookbook](command_cookbook.md)

## GPU Research Workflow

- [GPU quickstart](gpu_quickstart.md)
- [Colab L4 TinyStories v0.46.0 efficiency comparison notebook](../notebooks/colab_l4_v046_efficiency_comparison.ipynb)
- [Colab L4 Goal 48 code cached-sparse report notebook](../notebooks/colab_l4_goal48_code_cached_sparse_report.ipynb)
- [GPU job profiles](gpu_job_profiles.md)
- [GPU efficiency benchmarking](gpu_efficiency_benchmarking.md)
- [Warm sparse GPU efficiency comparison template](warm_sparse_efficiency_comparison_template.md)
- [Goal 46 GPU efficiency report](../reports/goal46_gpu_efficiency/README.md)
- [v0.46.0 L4 warm sparse comparison report](../reports/v0_46_0_l4_warm_sparse_comparison/README.md)
- [Goal 48 code cached-sparse L4 report](../reports/goal48_code_cached_sparse_efficiency/README.md)
- [GPU runtime limitations](gpu_runtime_limitations.md)
- [Serious jobs checklist](serious_jobs_checklist.md)
- [Colab 100M training notebook](colab_training.md)

## Implementation And Release Context

- [Config templates](config_templates.md)
- [Examples guide](examples_guide.md)
- [Known limitations](known_limitations.md)
- [Research positioning](research_positioning.md)
- [Release checklist](release_checklist.md)

## Current Evidence

The repository GPU evidence includes the Goal 46 report, the v0.46.0 L4 warm
sparse TinyStories comparison, and the Goal 48 code cached-sparse L4 report.
Together they show that MoP-Forge can measure Dense, MoP Full, adapter-only,
warm sparse, and cached sparse runs on Colab/L4 hardware.

These reports do not prove MoP superiority. The measured result is more careful:

- MoP Full matched Dense loss but was not more efficient.
- MoP Adapter-Only was much lighter and faster but had worse eval loss.
- Warm Adapter Norm/Head 64 and Warm LoRA Rank 8 looked stronger on the short
  TinyStories warm sparse run, but that evidence still needs longer runs and
  repeated seeds.
- Goal 48 cached sparse code training was much faster and used much less peak
  reserved VRAM than Dense in the measured cached-tail phase, but generated-code
  quality and time-to-target-loss were not proven in that report.
- The framework can measure the tradeoff and preserve the evidence.

## Current Implementation Focus

The current code adds the pieces needed for a more serious next comparison:

- warm-started sparse training,
- trainable-only sparse checkpoints,
- frozen-prefix activation caches,
- cached teacher top-k distillation for code sparse-tail training,
- optional hard-example replay from cached teacher CE loss,
- cached sparse offload of unused frozen backbone modules,
- best eval-loss checkpoint and time/tokens-to-target-loss reporting,
- fixed train/eval/test splits,
- generated-code quality metrics,
- routed FFN expert execution,
- internal routed low-rank deltas,
- comparison and acceptance-gate reports.

The next report should repeat the fixed code-dataset comparison with a configured
shared `target_eval_loss`, hard-example replay when justified by teacher-loss
metadata, and generated-code quality checks that can support or reject a
same-quality efficiency claim. Run longer GPU experiments before turning these
capabilities into broad performance claims.
