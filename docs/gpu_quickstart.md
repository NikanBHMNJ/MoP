# GPU Quickstart

MoP-Forge now includes a serious single-GPU research beta for tiny-to-small MoP
experiments and validated large-job profiles. It is not yet a fully production
distributed LLM training framework.

CPU-only development path:

```bash
mopforge runtime detect
mopforge gpu validate configs/jobs/tiny_gpu_smoke.json
mopforge gpu estimate configs/jobs/tiny_gpu_smoke.json
mopforge gpu train configs/jobs/tiny_gpu_smoke.json
```

CUDA path, when PyTorch CUDA is installed:

```bash
mopforge runtime detect
mopforge gpu train configs/jobs/tiny_gpu_smoke.json --device cuda --precision bf16
```

The tiny smoke profile falls back to CPU when CUDA is unavailable. Real GPU
performance is only validated on the user's hardware.

Useful follow-up commands:

```bash
mopforge gpu list
mopforge gpu show <run_id>
mopforge gpu resume <run_id>
mopforge gpu benchmark <run_id>
```
