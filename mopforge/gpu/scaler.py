"""AMP GradScaler wrapper for GPU training."""

from __future__ import annotations

from typing import Any

from mopforge.runtime import RuntimeContext


class AmpScaler:
    """Small wrapper that enables GradScaler only for CUDA fp16."""

    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self._scaler = None
        self._enabled = (
            runtime.device_info.device_type == "cuda"
            and runtime.precision_policy.selected == "fp16"
            and runtime.precision_policy.amp_enabled
        )
        if self._enabled:
            try:
                import torch

                scaler_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
                if scaler_cls is not None:
                    self._scaler = scaler_cls("cuda", enabled=True)
                else:
                    self._scaler = torch.cuda.amp.GradScaler(enabled=True)
            except Exception as exc:
                self._enabled = False
                runtime.warnings.append(f"GradScaler requested but unavailable: {exc}")

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._scaler is not None)

    def scale(self, loss):
        return self._scaler.scale(loss) if self.enabled else loss

    def step(self, optimizer) -> None:
        if self.enabled:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        if self.enabled:
            self._scaler.update()

    def unscale_(self, optimizer) -> None:
        if self.enabled:
            self._scaler.unscale_(optimizer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state": self._scaler.state_dict() if self.enabled else {},
            "selected_precision": self.runtime.precision_policy.selected,
            "device_type": self.runtime.device_info.device_type,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.enabled and isinstance(state, dict):
            self._scaler.load_state_dict(dict(state.get("state", {})))
