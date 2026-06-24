"""Packed fixed-length token shards for high-throughput causal LM training."""

from __future__ import annotations

import bisect
import hashlib
import json
import mmap
import struct
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from mopforge.tokenization import TokenizerSpec, build_tokenizer


TOKEN_SHARD_FORMAT = "mopforge_packed_tokens_v1"


@dataclass(slots=True)
class TokenShardBuildConfig:
    source_paths: list[str]
    tokenizer_spec_path: str
    output_dir: str
    sequence_length: int = 1024
    tokens_per_shard: int = 10_000_000
    eval_fraction: float = 0.01
    split_seed: int = 42
    text_field: str = "text"
    max_records: int | None = None
    drop_remainder: bool = True

    def __post_init__(self) -> None:
        if not self.source_paths:
            raise ValueError("source_paths must not be empty.")
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be greater than one.")
        if self.tokens_per_shard < self.sequence_length:
            raise ValueError("tokens_per_shard must be at least sequence_length.")
        if not 0.0 < float(self.eval_fraction) < 1.0:
            raise ValueError("eval_fraction must be in (0, 1).")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive or None.")


class PackedTokenDataset:
    """Random-access dataset over memory-mapped uint32 token shards."""

    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_token_shard_manifest(self.manifest_path)
        if split not in {"train", "eval"}:
            raise ValueError("split must be train or eval.")
        self.split = split
        self.sequence_length = int(self.manifest["sequence_length"])
        self.shards = list(self.manifest["splits"].get(split) or [])
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["sequences"])
            self.cumulative.append(total)
        self._handles: dict[int, Any] = {}
        self._maps: dict[int, mmap.mmap] = {}

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - previous
        mapped = self._mapped_shard(shard_index)
        byte_offset = local_index * self.sequence_length * 4
        raw = mapped[byte_offset : byte_offset + self.sequence_length * 4]
        values = struct.unpack(f"<{self.sequence_length}I", raw)
        input_ids = torch.tensor(values, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones(self.sequence_length, dtype=torch.long),
            "target_modules": [],
            "source_id": f"{self.split}:{index}",
        }

    def close(self) -> None:
        for mapped in self._maps.values():
            mapped.close()
        for handle in self._handles.values():
            handle.close()
        self._maps.clear()
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        state["_maps"] = {}
        return state

    def _mapped_shard(self, index: int) -> mmap.mmap:
        if index in self._maps:
            return self._maps[index]
        path = self.manifest_path.parent / self.shards[index]["path"]
        handle = path.open("rb")
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        self._handles[index] = handle
        self._maps[index] = mapped
        return mapped


def build_token_shards(config: TokenShardBuildConfig) -> dict[str, Any]:
    """Tokenize, split by document, pack, and write reproducible token shards."""

    tokenizer_spec = TokenizerSpec.load_json(config.tokenizer_spec_path)
    tokenizer = build_tokenizer(tokenizer_spec)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Packed token shards require a tokenizer EOS token.")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writers = {
        split: _ShardWriter(
            output,
            split=split,
            sequence_length=config.sequence_length,
            tokens_per_shard=config.tokens_per_shard,
        )
        for split in ("train", "eval")
    }
    buffers = {"train": [], "eval": []}
    records = 0
    raw_tokens = 0
    for source_id, text in _iter_source_text(config):
        split = _document_split(source_id, config.split_seed, config.eval_fraction)
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        token_ids.append(int(eos_token_id))
        raw_tokens += len(token_ids)
        records += 1
        buffer = buffers[split]
        buffer.extend(token_ids)
        while len(buffer) >= config.sequence_length:
            writers[split].add_sequence(buffer[: config.sequence_length])
            del buffer[: config.sequence_length]
    if not config.drop_remainder:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Keeping packed remainders requires a tokenizer pad token.")
        for split, buffer in buffers.items():
            if buffer:
                sequence = buffer + [int(pad_token_id)] * (config.sequence_length - len(buffer))
                writers[split].add_sequence(sequence)
    split_payload = {split: writer.finish() for split, writer in writers.items()}
    if not split_payload["train"] or not split_payload["eval"]:
        raise ValueError(
            "Packed dataset produced an empty train or eval split; add records or increase eval_fraction."
        )
    manifest = {
        "format": TOKEN_SHARD_FORMAT,
        "sequence_length": config.sequence_length,
        "dtype": "uint32_le",
        "tokenizer_spec": tokenizer_spec.to_dict(),
        "tokenizer_spec_sha256": _file_sha256(Path(config.tokenizer_spec_path)),
        "build_config": asdict(config),
        "source_sha256": {
            str(Path(path)): _file_sha256(Path(path))
            for path in config.source_paths
        },
        "records": records,
        "raw_tokens": raw_tokens,
        "packed_tokens": sum(
            int(shard["tokens"])
            for shards in split_payload.values()
            for shard in shards
        ),
        "splits": split_payload,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def load_token_shard_manifest(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("format") != TOKEN_SHARD_FORMAT:
        raise ValueError(f"Unsupported token shard manifest: {candidate}")
    for split, shards in (payload.get("splits") or {}).items():
        for shard in shards:
            shard_path = candidate.parent / shard["path"]
            if not shard_path.is_file():
                raise FileNotFoundError(f"Token shard is missing: {shard_path}")
    return payload


def build_packed_token_dataloaders(
    manifest_path: str | Path,
    *,
    micro_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle_train: bool,
    shuffle_seed: int,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
):
    torch = _require_torch()
    train_dataset = PackedTokenDataset(manifest_path, "train")
    eval_dataset = PackedTokenDataset(manifest_path, "eval")
    train_sampler = None
    eval_sampler = None
    if distributed_world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=shuffle_train,
            seed=shuffle_seed,
        )
        eval_sampler = torch.utils.data.DistributedSampler(
            eval_dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=False,
        )
    generator = torch.Generator().manual_seed(int(shuffle_seed))
    loader = {
        "batch_size": micro_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=bool(shuffle_train and train_sampler is None),
        sampler=train_sampler,
        generator=generator if shuffle_train and train_sampler is None else None,
        **loader,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        sampler=eval_sampler,
        **loader,
    )
    manifest = train_dataset.manifest
    metadata = {
        "kind": "packed_token_shards",
        "source_path": str(Path(manifest_path)),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "record_count": int(manifest.get("records", 0)),
        "sequence_length": int(manifest["sequence_length"]),
        "raw_tokens": int(manifest.get("raw_tokens", 0)),
        "packed_tokens": int(manifest.get("packed_tokens", 0)),
        "packing_efficiency": (
            float(manifest.get("packed_tokens", 0)) / max(1, int(manifest.get("raw_tokens", 0)))
        ),
        "tokenizer_spec_sha256": manifest.get("tokenizer_spec_sha256"),
        "shuffle_train": bool(shuffle_train),
        "shuffle_seed": int(shuffle_seed),
        "distributed_rank": int(distributed_rank),
        "distributed_world_size": int(distributed_world_size),
    }
    return train_loader, eval_loader, metadata


class _ShardWriter:
    def __init__(self, root: Path, *, split: str, sequence_length: int, tokens_per_shard: int):
        self.root = root
        self.split = split
        self.sequence_length = sequence_length
        self.max_sequences = max(1, tokens_per_shard // sequence_length)
        self.buffer = array("I")
        self.sequence_count = 0
        self.shard_index = 0
        self.shards: list[dict[str, Any]] = []

    def add_sequence(self, values: list[int]) -> None:
        if len(values) != self.sequence_length:
            raise ValueError("Packed sequence has the wrong length.")
        if any(value < 0 or value > 0xFFFFFFFF for value in values):
            raise ValueError("Token IDs must fit uint32.")
        self.buffer.extend(values)
        self.sequence_count += 1
        if self.sequence_count >= self.max_sequences:
            self._flush()

    def finish(self) -> list[dict[str, Any]]:
        self._flush()
        return list(self.shards)

    def _flush(self) -> None:
        if not self.sequence_count:
            return
        filename = f"{self.split}-{self.shard_index:06d}.bin"
        path = self.root / filename
        values = self.buffer
        if values.itemsize != 4:
            raise RuntimeError("Platform uint32 array storage is unavailable.")
        import sys

        if sys.byteorder != "little":
            values.byteswap()
        with path.open("wb") as handle:
            values.tofile(handle)
        tokens = self.sequence_count * self.sequence_length
        self.shards.append(
            {
                "path": filename,
                "sequences": self.sequence_count,
                "tokens": tokens,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
        self.buffer = array("I")
        self.sequence_count = 0
        self.shard_index += 1


def _iter_source_text(config: TokenShardBuildConfig) -> Iterator[tuple[str, str]]:
    emitted = 0
    for source in config.source_paths:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Packed-token source does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if config.max_records is not None and emitted >= config.max_records:
                    return
                if not line.strip():
                    continue
                if path.suffix.lower() == ".jsonl":
                    record = json.loads(line)
                    value = record.get(config.text_field, record.get("content"))
                    if value is None:
                        raise ValueError(
                            f"Missing {config.text_field!r} in {path}:{line_number}."
                        )
                    source_id = str(record.get("id") or record.get("record_id") or f"{path}:{line_number}")
                    text = str(value)
                else:
                    source_id = f"{path}:{line_number}"
                    text = line.rstrip("\n")
                emitted += 1
                yield source_id, text


def _document_split(source_id: str, seed: int, eval_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "eval" if value < eval_fraction else "train"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for packed token datasets.") from exc
    return torch
