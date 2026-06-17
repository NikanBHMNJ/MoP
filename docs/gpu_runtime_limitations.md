# GPU Runtime Limitations

The GPU beta is intentionally conservative:

- No guaranteed 2B/7B training on all hardware.
- No custom CUDA kernels.
- No production FSDP or DeepSpeed integration.
- Torchrun launcher support is a dry-run/foundation layer.
- Multi-GPU training is not production hardened.
- Memory estimates are approximate.
- Job profiles require user-provided hardware and data.
- CPU fallback keeps tests reliable, but does not validate GPU throughput.
- MoP routing is improved at PyTorch metadata/planning level, not kernel-fused.
- Fast Parameters at scale remain experimental.
- FP8 remains planning/fallback only.

Use the profiles as starting points, then inspect runtime metadata, memory
snapshots, checkpoint outputs, and benchmark scaffolds from your own hardware.
