"""Cached frozen-prefix activation helpers for sparse GPU training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mopforge.models import adapter_names_from_target_modules, condition_names_from_target_modules
from mopforge.runtime import move_batch_to_device


CACHE_FORMAT = "mopforge_activation_cache_v1"


class CachedActivationDataset:
    """Dataset backed by cached hidden states before sparse trainable tails."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        hard_example_replay_enabled: bool = False,
        hard_example_replay_loss_threshold: float | None = None,
        hard_example_replay_multiplier: int = 1,
    ) -> None:
        self.path = Path(path)
        self.payload = load_activation_cache(self.path)
        splits = self.payload.get("splits", {})
        if split not in splits:
            raise ValueError(f"Activation cache split not found: {split}")
        self.split = split
        self.original_records = list(splits[split])
        self.records, self.replay_metadata = _apply_hard_example_replay(
            self.original_records,
            enabled=hard_example_replay_enabled and split == "train",
            loss_threshold=hard_example_replay_loss_threshold,
            multiplier=hard_example_replay_multiplier,
        )

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
    shuffle_train: bool = True,
    shuffle_seed: int = 42,
    hard_example_replay_enabled: bool = False,
    hard_example_replay_loss_threshold: float | None = None,
    hard_example_replay_multiplier: int = 1,
):
    """Build train/eval DataLoaders from an activation cache file."""

    torch = _require_torch()
    train_ds = CachedActivationDataset(
        path,
        split="train",
        hard_example_replay_enabled=hard_example_replay_enabled,
        hard_example_replay_loss_threshold=hard_example_replay_loss_threshold,
        hard_example_replay_multiplier=hard_example_replay_multiplier,
    )
    eval_ds = CachedActivationDataset(path, split="eval")
    loader_kwargs = {
        "batch_size": micro_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_cached_activations,
    }
    generator = torch.Generator()
    generator.manual_seed(int(shuffle_seed))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=shuffle_train,
        generator=generator if shuffle_train else None,
        **loader_kwargs,
    )
    eval_loader = torch.utils.data.DataLoader(eval_ds, shuffle=False, **loader_kwargs)
    metadata = {
        "kind": "activation_cache",
        "source_path": str(Path(path)),
        "cache_format": train_ds.payload.get("cache_format"),
        "sharded": bool(train_ds.payload.get("sharded")),
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "record_count": len(train_ds) + len(eval_ds),
        "original_train_examples": len(train_ds.original_records),
        "hard_example_replay": dict(train_ds.replay_metadata),
        "shuffle_train": bool(shuffle_train),
        "shuffle_seed": int(shuffle_seed),
        "cache_metadata": dict(train_ds.payload.get("metadata") or {}),
    }
    return train_loader, eval_loader, metadata


def collate_cached_activations(records: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _require_torch()
    batch = {
        "hidden_states": torch.stack([record["hidden_states"] for record in records], dim=0),
        "attention_mask": torch.stack([record["attention_mask"] for record in records], dim=0),
        "labels": torch.stack([record["labels"] for record in records], dim=0),
        "target_modules": [list(record.get("target_modules") or []) for record in records],
        "source_id": [record.get("source_id") for record in records],
        "cached_token_count": sum(int(record.get("token_count") or 0) for record in records),
    }
    if records and all("teacher_topk_indices" in record and "teacher_topk_logits" in record for record in records):
        batch["teacher_topk_indices"] = torch.stack([record["teacher_topk_indices"] for record in records], dim=0)
        batch["teacher_topk_logits"] = torch.stack([record["teacher_topk_logits"] for record in records], dim=0)
    if records and all("teacher_ce_loss" in record for record in records):
        batch["teacher_ce_loss"] = torch.tensor(
            [float(record["teacher_ce_loss"]) for record in records],
            dtype=torch.float32,
        )
    return batch


def write_activation_cache(
    *,
    model,
    train_loader,
    eval_loader,
    output_path: str | Path,
    runtime,
    dtype: str = "bf16",
    max_batches: int | None = None,
    teacher_top_k: int = 0,
    records_per_shard: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write frozen-prefix activations for train and eval splits."""

    torch = _require_torch()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if records_per_shard is not None and records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive or None.")
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
            teacher_top_k=teacher_top_k,
        ),
        "eval": _cache_split(
            model,
            eval_loader,
            runtime,
            dtype=dtype,
            max_batches=max_batches,
            split="eval",
            teacher_top_k=teacher_top_k,
        ),
    }
    if previous_mode:
        model.train()
    metadata_payload = {
        "dtype": dtype,
        "teacher_top_k": int(teacher_top_k),
        "distillation_ready": bool(teacher_top_k),
        "teacher_loss_ready": bool(teacher_top_k),
        "train_records": len(splits["train"]),
        "eval_records": len(splits["eval"]),
        **dict(metadata or {}),
    }
    payload = {
        "cache_format": CACHE_FORMAT,
        "splits": splits,
        "metadata": metadata_payload,
    }
    if records_per_shard:
        result = _write_sharded_cache(
            output,
            splits=splits,
            metadata=metadata_payload,
            records_per_shard=records_per_shard,
        )
        return result
    torch.save(payload, output)
    return {
        "path": str(output),
        "cache_format": CACHE_FORMAT,
        "train_records": len(splits["train"]),
        "eval_records": len(splits["eval"]),
        "metadata": dict(payload["metadata"]),
        "sharded": False,
        "shard_count": 0,
    }


def load_activation_cache(path: str | Path) -> dict[str, Any]:
    torch = _require_torch()
    candidate = Path(path)
    if candidate.suffix.lower() == ".json":
        return _load_sharded_cache_manifest(candidate)
    try:
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(candidate, map_location="cpu")
    except Exception as torch_exc:
        try:
            return _load_sharded_cache_manifest(candidate)
        except Exception:
            raise torch_exc
    if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported activation cache format: {path}")
    return payload


def _write_sharded_cache(
    output: Path,
    *,
    splits: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    records_per_shard: int,
) -> dict[str, Any]:
    torch = _require_torch()
    shard_dir = output.parent / f"{output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits: dict[str, list[dict[str, Any]]] = {}
    shard_count = 0
    for split, records in splits.items():
        split_shards = []
        for shard_index, start in enumerate(range(0, len(records), records_per_shard)):
            chunk = records[start : start + records_per_shard]
            shard_path = shard_dir / f"{split}-{shard_index:06d}.pt"
            torch.save(
                {
                    "cache_format": CACHE_FORMAT,
                    "split": split,
                    "records": chunk,
                    "metadata": {
                        "record_count": len(chunk),
                        "records_per_shard": records_per_shard,
                    },
                },
                shard_path,
            )
            shard_count += 1
            split_shards.append(
                {
                    "path": _relative_to_parent(shard_path, output.parent),
                    "record_count": len(chunk),
                    "sha256": file_sha256(shard_path),
                }
            )
        manifest_splits[split] = split_shards
    manifest_metadata = {
        **dict(metadata),
        "sharded": True,
        "records_per_shard": records_per_shard,
        "shard_count": shard_count,
        "shard_dir": _relative_to_parent(shard_dir, output.parent),
    }
    manifest = {
        "cache_format": CACHE_FORMAT,
        "sharded": True,
        "splits": manifest_splits,
        "metadata": manifest_metadata,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "path": str(output),
        "cache_format": CACHE_FORMAT,
        "train_records": len(splits.get("train", [])),
        "eval_records": len(splits.get("eval", [])),
        "metadata": dict(manifest_metadata),
        "sharded": True,
        "shard_count": shard_count,
        "shard_dir": str(shard_dir),
    }


def _load_sharded_cache_manifest(path: Path) -> dict[str, Any]:
    torch = _require_torch()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("cache_format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported activation cache manifest: {path}")
    if not manifest.get("sharded"):
        raise ValueError(f"Activation cache manifest is not sharded: {path}")
    loaded_splits: dict[str, list[dict[str, Any]]] = {}
    shard_manifest = dict(manifest.get("splits") or {})
    for split, shards in shard_manifest.items():
        records: list[dict[str, Any]] = []
        for shard in shards:
            shard_path = path.parent / str(shard["path"])
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
                raise ValueError(f"Unsupported activation cache shard: {shard_path}")
            if payload.get("split") != split:
                raise ValueError(f"Activation cache shard split mismatch: {shard_path}")
            records.extend(list(payload.get("records") or []))
        loaded_splits[split] = records
    metadata = dict(manifest.get("metadata") or {})
    metadata.update(
        {
            "manifest_path": str(path),
            "manifest_sha256": file_sha256(path),
        }
    )
    return {
        "cache_format": CACHE_FORMAT,
        "sharded": True,
        "shards": shard_manifest,
        "splits": loaded_splits,
        "metadata": metadata,
    }


def _relative_to_parent(path: Path, parent: Path) -> str:
    try:
        return str(path.relative_to(parent))
    except ValueError:
        return str(path)


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
    teacher_top_k: int = 0,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    if teacher_top_k < 0:
        raise ValueError("teacher_top_k must be non-negative.")
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
            teacher_indices = None
            teacher_logits = None
            teacher_ce_loss = None
            if teacher_top_k:
                teacher = _teacher_topk_from_hidden(
                    model,
                    encoded["hidden_states"],
                    attention_mask=batch.get("attention_mask"),
                    labels=labels,
                    target_modules=target_modules,
                    top_k=teacher_top_k,
                    dtype=target_dtype,
                )
                teacher_indices = teacher["indices"]
                teacher_logits = teacher["logits"]
                teacher_ce_loss = teacher["ce_loss"]
            for row_index in range(hidden.shape[0]):
                row_mask = attention_mask[row_index].detach().cpu()
                row_labels = labels[row_index].detach().cpu()
                modules = (
                    list(target_modules[row_index])
                    if row_index < len(target_modules)
                    else []
                )
                record = {
                    "hidden_states": hidden[row_index],
                    "attention_mask": row_mask,
                    "labels": row_labels,
                    "target_modules": modules,
                    "source_id": f"{split}:{batch_index}:{row_index}",
                    "token_count": int(row_mask.sum().detach().cpu().item()),
                }
                if teacher_indices is not None and teacher_logits is not None:
                    record["teacher_topk_indices"] = teacher_indices[row_index]
                    record["teacher_topk_logits"] = teacher_logits[row_index]
                if teacher_ce_loss is not None:
                    record["teacher_ce_loss"] = float(teacher_ce_loss[row_index])
                records.append(record)
    return records


def _teacher_topk_from_hidden(
    model,
    hidden_states,
    *,
    attention_mask,
    labels,
    target_modules,
    top_k: int,
    dtype,
) -> dict[str, Any]:
    torch = _require_torch()
    vocab_size = int(getattr(model, "vocab_size", 0) or 0)
    k = int(min(max(1, top_k), vocab_size)) if vocab_size else int(max(1, top_k))
    active_adapters = None
    active_conditions = None
    if getattr(model, "fast_adapter_bank", None) is not None:
        active_adapters = [adapter_names_from_target_modules(item) for item in (target_modules or [])]
    if getattr(model, "generated_adapter", None) is not None:
        active_conditions = [condition_names_from_target_modules(item) for item in (target_modules or [])]
    outputs = model.forward_from_hidden(
        hidden_states,
        attention_mask=attention_mask,
        labels=labels,
        active_modules=target_modules,
        active_adapters=active_adapters,
        active_conditions=active_conditions,
    )
    logits = outputs["logits"].detach().float()
    topk = torch.topk(logits, k=k, dim=-1)
    return {
        "indices": topk.indices.detach().cpu().to(dtype=torch.long),
        "logits": topk.values.detach().cpu().to(dtype=dtype),
        "ce_loss": _per_example_ce_loss(logits, labels).detach().cpu(),
    }


def _per_example_ce_loss(logits, labels):
    torch = _require_torch()
    if labels is None:
        return torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous().to(device=logits.device)
    flat_loss = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shifted_labels.shape)
    mask = (shifted_labels != -100).to(dtype=flat_loss.dtype)
    return (flat_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _apply_hard_example_replay(
    records: list[dict[str, Any]],
    *,
    enabled: bool,
    loss_threshold: float | None,
    multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if multiplier <= 0:
        raise ValueError("hard_example_replay_multiplier must be positive.")
    if loss_threshold is not None and loss_threshold < 0:
        raise ValueError("hard_example_replay_loss_threshold must be non-negative or None.")
    if not enabled:
        return list(records), {
            "enabled": False,
            "loss_threshold": loss_threshold,
            "multiplier": multiplier,
            "hard_example_count": 0,
            "replayed_example_count": 0,
        }
    hard_records = [
        record
        for record in records
        if _is_hard_example_record(record, loss_threshold=loss_threshold)
    ]
    extra_copies = max(0, int(multiplier) - 1)
    expanded = list(records)
    for _ in range(extra_copies):
        expanded.extend(dict(record) for record in hard_records)
    return expanded, {
        "enabled": True,
        "loss_threshold": loss_threshold,
        "multiplier": multiplier,
        "hard_example_count": len(hard_records),
        "replayed_example_count": len(hard_records) * extra_copies,
    }


def _is_hard_example_record(record: dict[str, Any], *, loss_threshold: float | None) -> bool:
    if bool(record.get("hard_example")):
        return True
    if loss_threshold is None:
        return False
    try:
        return float(record.get("teacher_ce_loss")) >= float(loss_threshold)
    except (TypeError, ValueError):
        return False


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
