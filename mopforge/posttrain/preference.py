"""Reference-efficient DPO/ORPO post-training for production checkpoints."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mopforge.formatting import format_lesson_for_causal_lm
from mopforge.kts import KnowledgeLesson
from mopforge.models import load_gpu_checkpoint_model


@dataclass(slots=True)
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    record_id: str | None = None
    reference_chosen_logp: float | None = None
    reference_rejected_logp: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("prompt", "chosen", "rejected"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected must differ.")
        if (self.reference_chosen_logp is None) != (
            self.reference_rejected_logp is None
        ):
            raise ValueError("Both cached reference log probabilities are required together.")
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceRecord":
        return cls(**dict(data))


@dataclass(slots=True)
class PreferenceTrainingConfig:
    checkpoint_path: str
    preference_path: str
    output_dir: str = "posttrain_runs/preference"
    config_path: str | None = None
    reference_checkpoint_path: str | None = None
    objective: str = "dpo"
    beta: float = 0.1
    orpo_lambda: float = 0.1
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_optimizer_steps: int = 100
    max_seq_len: int = 1024
    seed: int = 42
    device: str = "auto"
    precision: str = "bf16"
    max_grad_norm: float | None = 1.0
    save_optimizer_state: bool = False
    cached_preferences_path: str | None = None

    def __post_init__(self) -> None:
        if self.objective not in {"dpo", "orpo"}:
            raise ValueError("objective must be dpo or orpo.")
        for name in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_optimizer_steps",
            "max_seq_len",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("beta", "orpo_lambda", "learning_rate"):
            if not isinstance(getattr(self, name), (int, float)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "PreferenceTrainingConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**dict(raw.get("payload") or raw))


def load_preference_records(path: str | Path) -> list[PreferenceRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(PreferenceRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid preference record at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("Preference dataset is empty.")
    return records


def write_preference_records(
    records: list[PreferenceRecord], path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return output


def build_verified_preference_records(
    lesson_path: str | Path,
    generation_eval_path: str | Path,
) -> list[PreferenceRecord]:
    """Pair verified targets with model generations that failed or differed."""

    lessons: dict[str, KnowledgeLesson] = {}
    with Path(lesson_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lesson = KnowledgeLesson.from_dict(json.loads(line))
                lessons[lesson.id] = lesson
    payload = json.loads(Path(generation_eval_path).read_text(encoding="utf-8"))
    values = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Generation evaluation must be a list or contain results[].")
    records = []
    for item in values:
        lesson_id = str(item.get("lesson_id") or "")
        lesson = lessons.get(lesson_id)
        rejected = item.get("generated_text")
        if lesson is None or not isinstance(rejected, str) or not rejected.strip():
            continue
        formatted = format_lesson_for_causal_lm(lesson)
        chosen = str(formatted["target"])
        if rejected.strip() == chosen.strip():
            continue
        records.append(
            PreferenceRecord(
                record_id=lesson_id,
                prompt=str(formatted["prompt"]),
                chosen=chosen,
                rejected=rejected,
                metadata={
                    "candidate_passed": bool(item.get("passed")),
                    "candidate_exact_match": bool(item.get("exact_match")),
                    "failure_type": item.get("failure_type"),
                    "bug_type": item.get("bug_type"),
                },
            )
        )
    if not records:
        raise ValueError("No differing generated candidates could form preference pairs.")
    return records


def sequence_log_probs(logits, labels):
    """Return summed and mean response log probabilities for masked labels."""

    torch = _require_torch()
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    sums = (token_logps * mask).sum(dim=-1)
    counts = mask.sum(dim=-1).clamp_min(1)
    return sums, sums / counts


def dpo_loss(
    policy_chosen_logp,
    policy_rejected_logp,
    reference_chosen_logp,
    reference_rejected_logp,
    *,
    beta: float,
):
    torch = _require_torch()
    margin = (policy_chosen_logp - policy_rejected_logp) - (
        reference_chosen_logp - reference_rejected_logp
    )
    losses = -torch.nn.functional.logsigmoid(float(beta) * margin)
    return losses.mean(), margin.detach()


def orpo_loss(
    chosen_logp_sum,
    rejected_logp_sum,
    chosen_logp_mean,
    *,
    coefficient: float,
):
    torch = _require_torch()
    chosen = chosen_logp_sum.clamp(max=-1e-6)
    rejected = rejected_logp_sum.clamp(max=-1e-6)
    chosen_log_odds = chosen - torch.log1p(-torch.exp(chosen))
    rejected_log_odds = rejected - torch.log1p(-torch.exp(rejected))
    preference = -torch.nn.functional.logsigmoid(chosen_log_odds - rejected_log_odds)
    sft = -chosen_logp_mean
    return (sft + float(coefficient) * preference).mean(), preference.detach()


class PreferenceTrainer:
    """Single-accelerator production DPO/ORPO trainer with cached references."""

    def __init__(self, config: PreferenceTrainingConfig):
        self.config = config

    def train(self) -> dict[str, Any]:
        torch = _require_torch()
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        records = load_preference_records(self.config.preference_path)
        loaded = load_gpu_checkpoint_model(
            self.config.checkpoint_path,
            config_path=self.config.config_path,
        )
        tokenizer = loaded["tokenizer"]
        device = _resolve_device(torch, self.config.device)
        if self.config.objective == "dpo" and not all(
            record.reference_chosen_logp is not None for record in records
        ):
            reference_path = (
                self.config.reference_checkpoint_path or self.config.checkpoint_path
            )
            reference = load_gpu_checkpoint_model(
                reference_path,
                config_path=self.config.config_path,
            )["model"]
            records = cache_reference_log_probs(
                records,
                reference,
                tokenizer,
                device=device,
                max_seq_len=self.config.max_seq_len,
                batch_size=self.config.micro_batch_size,
            )
            del reference
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cached_path = self.config.cached_preferences_path or str(
                Path(self.config.output_dir) / "preferences_with_reference.jsonl"
            )
            write_preference_records(records, cached_path)
        dataset = PreferenceDataset(records, tokenizer, self.config.max_seq_len)
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda values: collate_preference_batch(
                values, tokenizer.pad_token_id
            ),
        )
        model = loaded["model"].to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and self.config.precision == "fp16",
        )
        optimizer.zero_grad(set_to_none=True)
        updates = 0
        microsteps = 0
        losses = []
        margins = []
        started = time.perf_counter()
        while updates < self.config.max_optimizer_steps:
            for batch in loader:
                batch = _move_batch(batch, device)
                with _autocast(torch, device, self.config.precision):
                    chosen = model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                    )
                    rejected = model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                    )
                    chosen_sum, chosen_mean = sequence_log_probs(
                        chosen["logits"], batch["chosen_labels"]
                    )
                    rejected_sum, _ = sequence_log_probs(
                        rejected["logits"], batch["rejected_labels"]
                    )
                    if self.config.objective == "dpo":
                        loss, margin = dpo_loss(
                            chosen_sum,
                            rejected_sum,
                            batch["reference_chosen_logp"],
                            batch["reference_rejected_logp"],
                            beta=self.config.beta,
                        )
                    else:
                        loss, margin = orpo_loss(
                            chosen_sum,
                            rejected_sum,
                            chosen_mean,
                            coefficient=self.config.orpo_lambda,
                        )
                scaler.scale(
                    loss / self.config.gradient_accumulation_steps
                ).backward()
                microsteps += 1
                losses.append(float(loss.detach().cpu()))
                margins.append(float(margin.float().mean().cpu()))
                if microsteps % self.config.gradient_accumulation_steps:
                    continue
                if self.config.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates >= self.config.max_optimizer_steps:
                    break
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from mopforge.gpu.checkpointing import save_gpu_checkpoint

        checkpoint = save_gpu_checkpoint(
            output / "final.pt",
            model=model,
            optimizer=optimizer if self.config.save_optimizer_state else None,
            state={
                "global_step": microsteps,
                "optimizer_step": updates,
                "tokens_seen": 0,
            },
            config=loaded["config"],
            model_metadata={
                "architecture": loaded["architecture"].to_dict(),
                "posttraining": self.config.to_dict(),
            },
        )
        result = {
            "format": "mopforge_preference_training_v1",
            "objective": self.config.objective,
            "records": len(records),
            "optimizer_steps": updates,
            "microsteps": microsteps,
            "final_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "mean_preference_margin": sum(margins) / len(margins),
            "duration_seconds": time.perf_counter() - started,
            "checkpoint": checkpoint,
            "config": self.config.to_dict(),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result


class PreferenceDataset:
    def __init__(self, records, tokenizer, max_seq_len):
        self.items = [
            _encode_record(record, tokenizer, max_seq_len) for record in records
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_preference_batch(items, pad_token_id):
    torch = _require_torch()
    output = {}
    for prefix in ("chosen", "rejected"):
        maximum = max(len(item[f"{prefix}_input_ids"]) for item in items)
        ids, masks, labels = [], [], []
        for item in items:
            padding = maximum - len(item[f"{prefix}_input_ids"])
            ids.append(item[f"{prefix}_input_ids"] + [pad_token_id] * padding)
            masks.append([1] * len(item[f"{prefix}_input_ids"]) + [0] * padding)
            labels.append(item[f"{prefix}_labels"] + [-100] * padding)
        output[f"{prefix}_input_ids"] = torch.tensor(ids, dtype=torch.long)
        output[f"{prefix}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        output[f"{prefix}_labels"] = torch.tensor(labels, dtype=torch.long)
    if all(item["reference_chosen_logp"] is not None for item in items):
        output["reference_chosen_logp"] = torch.tensor(
            [item["reference_chosen_logp"] for item in items], dtype=torch.float32
        )
        output["reference_rejected_logp"] = torch.tensor(
            [item["reference_rejected_logp"] for item in items], dtype=torch.float32
        )
    return output


def cache_reference_log_probs(
    records,
    model,
    tokenizer,
    *,
    device,
    max_seq_len,
    batch_size,
):
    torch = _require_torch()
    dataset = PreferenceDataset(records, tokenizer, max_seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda values: collate_preference_batch(values, tokenizer.pad_token_id),
    )
    model = model.to(device)
    model.eval()
    chosen_values, rejected_values = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            with _autocast(torch, device, "bf16" if device.type == "cuda" else "fp32"):
                chosen = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                )
                rejected = model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                )
                chosen_sum, _ = sequence_log_probs(
                    chosen["logits"], batch["chosen_labels"]
                )
                rejected_sum, _ = sequence_log_probs(
                    rejected["logits"], batch["rejected_labels"]
                )
            chosen_values.extend(chosen_sum.cpu().tolist())
            rejected_values.extend(rejected_sum.cpu().tolist())
    return [
        PreferenceRecord(
            **{
                **record.to_dict(),
                "reference_chosen_logp": float(chosen),
                "reference_rejected_logp": float(rejected),
            }
        )
        for record, chosen, rejected in zip(
            records, chosen_values, rejected_values, strict=True
        )
    ]


def _encode_record(record, tokenizer, max_seq_len):
    prompt = tokenizer.encode(record.prompt, add_special_tokens=False)
    chosen = tokenizer.encode(record.chosen, add_special_tokens=False)
    rejected = tokenizer.encode(record.rejected, add_special_tokens=False)
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]

    def encode_response(response):
        if max_seq_len <= len(bos) + len(eos):
            raise ValueError("max_seq_len is too small for tokenizer special tokens.")
        if len(bos) + len(response) + len(eos) >= max_seq_len:
            response = response[: max_seq_len - len(bos) - len(eos)]
            prompt_part = []
        else:
            prompt_budget = max_seq_len - len(bos) - len(response) - len(eos)
            prompt_part = prompt[-prompt_budget:] if prompt_budget else []
        ids = bos + prompt_part + response + eos
        labels = [-100] * (len(bos) + len(prompt_part)) + response + eos
        return ids, labels

    chosen_ids, chosen_labels = encode_response(chosen)
    rejected_ids, rejected_labels = encode_response(rejected)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "reference_chosen_logp": record.reference_chosen_logp,
        "reference_rejected_logp": record.reference_rejected_logp,
    }


def _move_batch(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def _resolve_device(torch, value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _autocast(torch, device, precision):
    from contextlib import nullcontext

    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for preference post-training.") from exc
    return torch
