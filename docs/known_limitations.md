# Known Limitations

- MoP-Forge has a DDP/FSDP training beta, not a managed or hardware-proven
  production training service.
- No DeepSpeed, elastic recovery, remote object store, or cluster scheduler is
  implemented.
- The CLI prints torchrun commands but does not submit cluster jobs.
- No custom CUDA kernels are implemented.
- MoP routing is PyTorch-level metadata/grouping, not kernel-fused routing.
- FP8 remains planning-only.
- No guaranteed 2B/7B training on all hardware.
- The memory estimator is approximate and should be treated as a planning tool.
- Job profiles require user-provided hardware and local data.
- CPU fallback exists for tests, but it does not validate real GPU performance.
- Fast Parameters at scale are experimental.
- The local Python verifier is not sandboxed.
- DPO/ORPO post-training is single-accelerator; distributed SFT uses
  `GPUTrainer`.
- Hugging Face export materializes one MoP expert as a standard dense Llama;
  it does not preserve dynamic MoP routing in Transformers.
- No external dataset downloads, model downloads, cloud launcher, or remote
  object store are included.
