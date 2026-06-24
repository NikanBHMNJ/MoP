# Goal 51 1B A100 Feasibility Probe

This directory is the lightweight report target for the Goal 51 admission run.
No A100 probe has been measured or committed yet.

Run `notebooks/colab_a100_goal51_1b_feasibility_probe.ipynb` on an A100 40 GB or
80 GB. The notebook selects the matching config and exports:

- `gpu_probe_report.json`
- the exact probe config
- a short measured README

Checkpoints, activation caches, optimizer state, and model weights must not be
added here. A 1B pilot is admitted only when the measured report passes all
memory, finite/decreasing-loss, no-OOM, and checkpoint-resume gates.

The implemented profiles are:

- `configs/jobs/1b_dense_a100_40gb_probe.json`
- `configs/jobs/1b_dense_a100_80gb_probe.json`
- `configs/jobs/1b_mop_full_a100_40gb_probe.json`
- `configs/jobs/1b_mop_full_a100_80gb_probe.json`
- `configs/jobs/1b_cached_adapter_128_a100_40gb_probe.json`
- `configs/jobs/1b_cached_adapter_128_a100_80gb_probe.json`
