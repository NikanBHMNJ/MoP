"""Local BPE tokenizer training with reproducible corpus metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from mopforge.tokenization.base import TokenizerSpec


@dataclass(slots=True)
class BPETrainingConfig:
    source_paths: list[str]
    output_dir: str
    vocab_size: int = 32768
    min_frequency: int = 2
    text_field: str = "text"
    special_tokens: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
    max_records: int | None = None

    def __post_init__(self) -> None:
        if not self.source_paths:
            raise ValueError("source_paths must not be empty.")
        if self.vocab_size < len(self.special_tokens) + 256:
            raise ValueError("vocab_size is too small for byte-level BPE.")
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive.")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive or None.")


def train_bpe_tokenizer(config: BPETrainingConfig) -> dict:
    """Train and save a byte-level BPE tokenizer from text or JSONL files."""

    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    except ImportError as exc:
        raise ImportError(
            "BPE training requires the optional `tokenizers` dependency; install `mopforge[hf]`."
        ) from exc

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    counter = {"records": 0, "characters": 0}

    def iterator():
        for text in _iter_text(config):
            counter["records"] += 1
            counter["characters"] += len(text)
            yield text

    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", bos), ("<eos>", eos)],
    )
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    spec = TokenizerSpec(
        tokenizer_type="hf",
        name_or_path=str(tokenizer_path),
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<pad>"),
        bos_token_id=bos,
        eos_token_id=eos,
        unk_token_id=tokenizer.token_to_id("<unk>"),
        metadata={
            "backend": "tokenizers",
            "algorithm": "byte_level_bpe",
            "source_sha256": {
                str(Path(path)): _file_sha256(Path(path))
                for path in config.source_paths
            },
            "training_config": asdict(config),
        },
    )
    spec_path = spec.save_json(output / "tokenizer_spec.json")
    report = {
        "format": "mopforge_bpe_training_v1",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_spec_path": str(spec_path),
        "vocab_size": tokenizer.get_vocab_size(),
        **counter,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _iter_text(config: BPETrainingConfig) -> Iterator[str]:
    emitted = 0
    for source in config.source_paths:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer source does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            values = _iter_jsonl(path, config.text_field)
        else:
            values = _iter_plain_text(path)
        for text in values:
            if config.max_records is not None and emitted >= config.max_records:
                return
            if text:
                emitted += 1
                yield text


def _iter_jsonl(path: Path, text_field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get(text_field)
            if value is None:
                value = record.get("content")
            if value is None:
                raise ValueError(
                    f"Missing text field {text_field!r} in {path}:{line_number}."
                )
            yield str(value)


def _iter_plain_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line.rstrip("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
