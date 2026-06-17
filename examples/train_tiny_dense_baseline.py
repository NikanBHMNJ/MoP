"""Run a tiny CPU smoke-training loop over generated lessons."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import CausalLMCollator, LessonCausalLMDataset
from mopforge.kts import LessonStore
from mopforge.models import TinyCausalTransformer
from mopforge.tokenization import ByteTokenizer


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if DATA_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(DATA_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Run a short deterministic smoke-training loop."""

    if CausalLMCollator is None or TinyCausalTransformer is None:
        print("PyTorch is not installed; skipping tiny dense baseline smoke run.")
        return

    import torch
    from torch.utils.data import DataLoader

    ensure_lessons()
    torch.manual_seed(7)

    lessons = LessonStore(DATA_PATH).load_all()
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=512)
    collator = CausalLMCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    model = TinyCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=512,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss={loss.item():.4f}")
        if step >= 3:
            break

    print("Tiny dense baseline smoke run complete. This is not meaningful training.")


if __name__ == "__main__":
    main()
