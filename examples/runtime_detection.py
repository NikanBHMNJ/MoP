"""Inspect runtime/device support without requiring CUDA."""

from __future__ import annotations

import json

from mopforge.runtime import RuntimeConfig, build_runtime_context, detect_devices, runtime_metadata


def dry_run(config: RuntimeConfig) -> dict:
    try:
        return runtime_metadata(build_runtime_context(config))
    except Exception as exc:
        return {
            "requested_device": config.device,
            "requested_precision": config.precision,
            "error": str(exc),
        }


def main() -> None:
    print("Runtime detection smoke only. CUDA is optional.")
    payload = {
        "detect": detect_devices(),
        "cpu_fp32": dry_run(RuntimeConfig(device="cpu", precision="fp32")),
        "auto_auto": dry_run(RuntimeConfig(device="auto", precision="auto", enable_amp=True, require_device_available=False)),
        "cuda_bf16_plan": dry_run(RuntimeConfig(device="cuda", precision="bf16", enable_amp=True, allow_tf32=True, require_device_available=False)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
