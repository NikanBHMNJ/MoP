"""Cached frozen-prefix activation helpers for sparse GPU training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mopforge.runtime import move_batch_to_device


CACHE_FORMAT = "mopforge_activation_cache_v1"


class CachedActivationDataset:
    """Dataset backed by cached hidden states before sparse trainable tails."""

    def __init__(self, path: str | Path, *, split: str = "train") -> None:
        self.path = Path(path)
        self.payload = load_activation_cache(self.path)
        splits = self.payload.get("splits", {})
        if split not in splits:
            raise ValueError(f"Activation cache split not found: {split}")
        self.split = split
        self.records = list(splits[split])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])


def build_cached_activation_dataloaders(
    path: str | Path,
    *,
    micro_batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Build train/eval DataLoaders from an activation cache file."""

    torch = _require_torch()
    train_ds = CachedActivationDataset(path, split="train")
    eval_ds = CachedActivationDataset(path, split="eval")
    loader_kwargs = {
        "batch_size": micro_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_cached_activations,
    }
    train_loader = torch.utils.data.DataLoader(train_ds, **loader_kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_ds, **loader_kwargs)
    metadata = {
        "kind": "activation_cache",
        "source_path": str(Path(path)),
        "cache_format": train_ds.payload.get("cache_format"),
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "record_count": len(train_ds) + len(eval_ds),
        "cache_metadata": dict(train_ds.payload.get("metadata") or {}),
    }
    return train_loader, eval_loader, metadata


def collate_cached_activations(records: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _require_torch()
    return {
        "hidden_states": torch.stack([record["hidden_states"] for record in records], dim=0),
        "attention_mask": torch.stack([record["attention_mask"] for record in records], dim=0),
        "labels": torch.stack([record["labels"] for record in records], dim=0),
        "target_modules": [list(record.get("target_modules") or []) for record in records],
        "source_id": [record.get("source_id") for record in records],
        "cached_token_count": sum(int(record.get("token_count") or 0) for record in records),
    }


def write_activation_cache(
    *,
    model,
    train_loader,
    eval_loader,
    output_path: str | Path,
    runtime,
    dtype: str = "bf16",
    max_batches: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write frozen-prefix activations for train and eval splits."""

    torch = _require_torch()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = model.training
    model.eval()
    splits = {
        "train": _cache_split(
            model,
            train_loader,
            runtime,
            dtype=dtype,
            max_batches=max_batches,
            split="train",
        ),
        "eval": _cache_split(
            model,
            eval_loader,
            runtime,
            dtype=dtype,
            max_batches=max_batches,
            split="eval",
        ),
    }
    if previous_mode:
        model.train()
    payload = {
        "cache_format": CACHE_FORMAT,
        "splits": splits,
        "metadata": {
            "dtype": dtype,
            "train_records": len(splits["train"]),
            "eval_records": len(splits["eval"]),
            **dict(metadata or {}),
        },
    }
    torch.save(payload, output)
    return {
        "path": str(output),
        "cache_format": CACHE_FORMAT,
        "train_records": len(splits["train"]),
        "eval_records": len(splits["eval"]),
        "metadata": dict(payload["metadata"]),
    }


def load_activation_cache(path: str | Path) -> dict[str, Any]:
    torch = _require_torch()
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported activation cache format: {path}")
    return payload


def file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Any) -> str:
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_split(
    model,
    loader,
    runtime,
    *,
    dtype: str,
    max_batches: int | None,
    split: str,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    target_dtype = _dtype(dtype)
    records: list[dict[str, Any]] = []
    device = runtime.device_info.selected if runtime is not None else "cpu"
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            target_modules = batch.get("target_modules") or []
            encoded = model.encode_for_sparse_tail(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                active_modules=target_modules,
            )
            _validate_frozen_cache_boundary(model, encoded)
            hidden = encoded["hidden_states"].detach().cpu().to(dtype=target_dtype)
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")
            for row_index in range(hidden.shape[0]):
                row_mask = attention_mask[row_index].detach().cpu()
                row_labels = labels[row_index].detach().cpu()
                modules = (
                    list(target_modules[row_index])
                    if row_index < len(target_modules)
                    else []
                )
                records.append(
                    {
                        "hidden_states": hidden[row_index],
                        "attention_mask": row_mask,
                        "labels": row_labels,
                        "target_modules": modules,
                        "source_id": f"{split}:{batch_index}:{row_index}",
                        "token_count": int(row_mask.sum().detach().cpu().item()),
                    }
                )
    return records


def _validate_frozen_cache_boundary(model, encoded: dict[str, Any]) -> None:
    metadata = dict(encoded.get("metadata") or {})
    unsafe = []
    if not metadata.get("frozen_prefix_no_grad_enabled"):
        unsafe.append("embeddings/shared prefix")
    if getattr(model, "module_bank", None) is not None and not metadata.get(
        "frozen_module_bank_no_grad_enabled"
    ):
        unsafe.append("module bank")
    if len(getattr(model, "routed_blocks", [])) and not metadata.get(
        "frozen_routed_blocks_no_grad_enabled"
    ):
        unsafe.append("routed expert blocks")
    if unsafe:
        raise ValueError(
            "Activation caching requires every encoded component to be frozen; "
            f"trainable components found: {', '.join(unsafe)}."
        )


def _dtype(dtype: str):
    torch = _require_torch()
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp32, fp16, or bf16.")


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for activation caches.") from exc
    return torch
