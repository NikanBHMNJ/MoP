# GPU Runtime Limitations

The GPU beta is intentionally conservative:

- No guaranteed 2B/7B training on all hardware.
- No custom CUDA kernels.
- DDP/FSDP execution and DCP sharded resume are implemented, but have not yet
  been validated by a committed multi-H100 hardware report.
- `mopforge gpu launch-torchrun` prints a dry-run command; operators execute
  the displayed `torchrun` command on their own cluster.
- No DeepSpeed integration or elastic/multi-node failure recovery.
- Memory estimates are approximate.
- Job profiles require user-provided hardware and data.
- CPU fallback keeps tests reliable, but does not validate GPU throughput.
- MoP routing is improved at PyTorch metadata/planning level, not kernel-fused.
- Fast Parameters at scale remain experimental.
- FP8 remains planning/fallback only.
- Preference post-training currently runs on one accelerator; use supervised
  `GPUTrainer` for the distributed SFT stage.
- Hugging Face export requires a consolidated checkpoint. Use
  `mopforge gpu consolidate-checkpoint` after FSDP training.
- The Python code verifier is process-isolated but is not a secure sandbox.

Use the profiles as starting points, then inspect runtime metadata, memory
snapshots, checkpoint outputs, and benchmark scaffolds from your own hardware.
