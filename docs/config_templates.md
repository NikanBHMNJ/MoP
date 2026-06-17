# Config Templates

Every template is a local JSON config envelope. CPU-safe templates validate
with `mopforge config validate`. GPU job profiles validate with
`mopforge gpu validate` and estimate with `mopforge gpu estimate`.

| config name | kind | purpose | CPU-safe? | GPU required? | executes training? | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/examples/trainer_runtime_auto.json` | `trainer` | tiny trainer runtime smoke | yes | no | only if run | auto runtime fallback |
| `configs/examples/sft_*.json` | `sft` | supervised fine-tuning smoke modes | yes | no | only if run | full/module/adapter/generated |
| `configs/examples/cpt_cpu.json` | `pretrain` | tiny continued pretraining | yes | no | only if run | local corpus |
| `configs/examples/benchmark_*.json` | `benchmark` | local benchmark scaffolds | yes | no | only if run | loss/code/router/composite |
| `configs/examples/analysis_*.json` | `analysis` | report scaffolds | yes | no | no | source IDs can be added later |
| `configs/examples/dataset_*.json` | `dataset` | local dataset registry actions | yes | no | no training | needs local files to execute |
| `configs/examples/model_*.json` | `model` | local model registry manifests | yes | no | no | future large configs are registry-only |
| `configs/examples/manifest_*.json` | `manifest` | research run planning | yes | no | no | planning artifact |
| `configs/examples/ablation_*.json` | `ablation` | CPU ablation scaffolds | yes | no | only if run | sequential local runs |
| `configs/jobs/tiny_gpu_smoke.json` | `gpu_train` | tiny GPU/CPU-fallback smoke | yes | optional | yes if run | safe for CPU fallback |
| `configs/jobs/100m_dense_a100_smoke.json` | `gpu_train` | 100M dense A100 profile | no | recommended | yes if run | serious hardware profile |
| `configs/jobs/100m_mop_a100_smoke.json` | `gpu_train` | 100M MoP A100 profile | no | recommended | yes if run | serious hardware profile |
| `configs/jobs/500m_dense_vs_mop_h100.json` | `gpu_train` | 500M H100 comparison plan | no | yes | only explicit | planning/validation first |
| `configs/jobs/1b_mop_h100_bf16.json` | `gpu_train` | 1B H100 bf16 profile | no | yes | only explicit | validate before executing |
| `configs/jobs/2b_mop_a100_plan.json` | `gpu_train` | 2B A100 plan | no | yes | no by default | planning profile |
| `configs/jobs/7b_mop_h100_plan.json` | `gpu_train` | 7B H100 plan | no | yes | no by default | planning profile |
| `configs/jobs/multigpu_mop_torchrun_plan.json` | `gpu_train` | torchrun dry-run plan | no | yes | no | launcher foundation only |
