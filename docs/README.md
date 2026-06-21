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
- [Colab L4 Goal 49 verified-code quality report notebook](../notebooks/colab_l4_goal49_verified_code_quality_report.ipynb)
- [Colab L4 Goal 50 100M learning-gate notebook](../notebooks/colab_l4_goal50_100m_learning_gate.ipynb)
- [Colab L4 Goal 50 full 100M quality comparison notebook](../notebooks/colab_l4_goal50_100m_quality_comparison.ipynb)
- [GPU job profiles](gpu_job_profiles.md)
- [GPU efficiency benchmarking](gpu_efficiency_benchmarking.md)
- [Warm sparse GPU efficiency comparison template](warm_sparse_efficiency_comparison_template.md)
- [Goal 46 GPU efficiency report](../reports/goal46_gpu_efficiency/README.md)
- [v0.46.0 L4 warm sparse comparison report](../reports/v0_46_0_l4_warm_sparse_comparison/README.md)
- [Goal 48 code cached-sparse L4 report](../reports/goal48_code_cached_sparse_efficiency/README.md)
- [Goal 49 verified code-quality L4 report](../reports/goal49_verified_code_quality_efficiency/README.md)
- [Goal 50 100M learning-gate L4 report](../reports/goal50_100m_learning_gate/README.md)
- [Goal 50 full 100M quality comparison](../reports/goal50_100m_quality_comparison/README.md)
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
sparse TinyStories comparison, the Goal 48 code cached-sparse report, and the
Goal 49 verified code-quality report. Together they show that MoP-Forge can
measure Dense, MoP Full, adapter-only, warm sparse, and cached sparse runs on
Colab/L4 hardware.

These reports do not prove MoP superiority. The measured result is more careful:

- MoP Full matched Dense loss but was not more efficient.
- MoP Adapter-Only was much lighter and faster but had worse eval loss.
- Warm Adapter Norm/Head 64 and Warm LoRA Rank 8 looked stronger on the short
  TinyStories warm sparse run, but that evidence still needs longer runs and
  repeated seeds.
- Goal 48 cached sparse code training was much faster and used much less peak
  reserved VRAM than Dense in the measured cached-tail phase, but generated-code
  quality and time-to-target-loss were not proven in that report.
- Goal 49 Cached Adapter/Norm/Head 128 measured `87,518.63` tokens/sec and
  `0.0605 GB` peak reserved VRAM versus Dense at `9,346.82` tokens/sec and
  `1.8652 GB`. Its checkpoint was `7.66 MB` versus `987.15 MB`, and its best
  eval loss was `0.4685` versus `0.8022`.
- Goal 49 Cached Tail-Only LoRA Rank 8 measured `71,133.54` tokens/sec,
  `0.0820 GB` peak reserved VRAM, an `8.05 MB` checkpoint, and `0.4980` best
  eval loss.
- Both Goal 49 cached profiles reached `50%` syntax pass on 32 samples, but all
  profiles remained at `0%` exact match and verifier pass. This validates the
  efficiency path, not useful generated-code quality.
- Goal 49 derived the target loss after baseline training, so only student
  time-to-target values are present; no baseline time-to-target comparison is
  claimed.
- Goal 50 adds the experiment-integrity path needed to diagnose that quality
  result: seeded epoch reshuffling, full eval loss, best-checkpoint generation,
  bug-category-balanced samples, per-category failures, raw/XML ground-truth
  controls, and prompt/target truncation statistics.
- The initial Goal 50 run passed every protocol/control check but failed all
  generation-quality gates because generation inserted EOS after the prompt
  while training did not.
- After that boundary fix, the rerun reached `100%` train and held-out XML
  completion, syntax, verifier, and exact match, with `0.0000904` best eval
  loss. The memorization gate now passes and Phase C is allowed.
- The full Goal 50 comparison then passed its corrected acceptance gate. Cached
  Adapter/Norm/Head 128 reached `88.0%` verifier/exact match versus Dense at
  `82.4%`, with `8.35x` throughput and `31.70x` lower peak reserved VRAM.
- The report-only correction uses sequence-length evidence from an uncached
  profile on the shared fixed split because cached loaders omit that metadata;
  no measured run value changed.
- The framework can measure the tradeoff and preserve the evidence.

## Current Implementation Focus

The current code adds the pieces needed for a more serious next comparison:

- warm-started sparse training,
- trainable-only sparse checkpoints,
- frozen-prefix activation caches,
- cached teacher top-k distillation for code sparse-tail training,
- optional hard-example replay from cached teacher CE loss,
- cached sparse offload of unused frozen backbone modules,
- cache-compatible tail-only LoRA rank 8/16 training,
- verified fixed-code XML target framing for small code-repair quality runs,
- best eval-loss checkpoint and time/tokens-to-target-loss reporting,
- fixed train/eval/test splits,
- optional bug-type-stratified fixed splits and seeded epoch reshuffling,
- full held-out eval and best-checkpoint generated-code evaluation,
- train/held-out per-category quality metrics, ground-truth controls, and
  pre-truncation sequence-length statistics,
- routed FFN expert execution,
- internal routed low-rank deltas,
- comparison and acceptance-gate reports.

The next permitted run is a separate 1B L4 memory/throughput probe. The full
100M comparison passed its narrow repair-quality and cached-efficiency gates,
but this does not establish broad code generation or 1B feasibility.

The Goal 49 Colab notebook automates that comparison and creates a downloadable
lightweight report. For cached profiles, full-model generation happens after
the cached-tail VRAM metrics are captured, so quality evaluation does not
inflate the reported sparse-training memory peak.

The Goal 50 notebook creates a separate lightweight diagnostic report with the
exact best checkpoint, microstep and optimizer-update counts, full train/eval
generation, per-category evidence, and ground-truth controls.

Once that diagnostic passes, the full Goal 50 100M comparison notebook runs
Dense, MoP Full, Warm Adapter/Norm/Head 128, Cached Adapter/Norm/Head 128, and
Cached Tail-Only LoRA Rank 8 on a 10,000-lesson balanced split. It enforces a
shared target loss and writes explicit quality-plus-efficiency acceptance gates.
