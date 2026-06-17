"""RNG state capture/restore helpers for local CPU training."""

from __future__ import annotations

import random
from typing import Any


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, optional NumPy, and optional PyTorch RNG state."""

    state: dict[str, Any] = {
        "has_python": True,
        "python_state": random.getstate(),
        "has_numpy": False,
        "numpy_state": None,
        "has_torch": False,
        "torch_state": None,
        "has_cuda": False,
        "cuda_state": None,
    }
    try:
        import numpy as np
    except Exception:
        pass
    else:
        state["has_numpy"] = True
        state["numpy_state"] = np.random.get_state()

    try:
        import torch
    except Exception:
        return state

    state["has_torch"] = True
    state["torch_state"] = torch.get_rng_state()
    try:
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    state["has_cuda"] = has_cuda
    if has_cuda:
        try:
            state["cuda_state"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda_state"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state` when available."""

    if not isinstance(state, dict):
        return
    if state.get("has_python") and state.get("python_state") is not None:
        random.setstate(state["python_state"])

    if state.get("has_numpy") and state.get("numpy_state") is not None:
        try:
            import numpy as np
        except Exception:
            pass
        else:
            np.random.set_state(state["numpy_state"])

    if state.get("has_torch") and state.get("torch_state") is not None:
        try:
            import torch
        except Exception:
            return
        torch.set_rng_state(state["torch_state"])
        if state.get("has_cuda") and state.get("cuda_state") is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["cuda_state"])
            except Exception:
                return
