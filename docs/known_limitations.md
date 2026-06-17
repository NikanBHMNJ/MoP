# Known Limitations

- MoP-Forge is not a production distributed LLM trainer.
- No production FSDP or DeepSpeed integration is implemented.
- Torchrun support is a dry-run launcher foundation, not hardened multi-GPU
  training.
- No custom CUDA kernels are implemented.
- MoP routing is PyTorch-level metadata/grouping, not kernel-fused routing.
- FP8 remains planning-only.
- No guaranteed 2B/7B training on all hardware.
- The memory estimator is approximate and should be treated as a planning tool.
- Job profiles require user-provided hardware and local data.
- CPU fallback exists for tests, but it does not validate real GPU performance.
- Fast Parameters at scale are experimental.
- The local Python verifier is not sandboxed.
- No external dataset downloads, model downloads, cloud launcher, or remote
  object store are included.
