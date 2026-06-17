"""Run a CPU-only TinyMoP smoke-training loop over generated lessons."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import CausalLMCollator, LessonCausalLMDataset
from mopforge.kts import LessonStore
from mopforge.models import TinyMoPCausalTransformer
from mopforge.tokenization import ByteTokenizer


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if DATA_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(DATA_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Run three tiny CPU-only TinyMoP training steps."""

    if CausalLMCollator is None or TinyMoPCausalTransformer is None:
        print("PyTorch is not installed; skipping TinyMoP CPU smoke test.")
        return

    import torch
    from torch.utils.data import DataLoader

    print("TinyMoP CPU smoke test only. Losses are not meaningful.")
    ensure_lessons()
    torch.manual_seed(11)

    lessons = LessonStore(DATA_PATH).load_all()
    tokenizer = ByteTokenizer()
    dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=512)
    collator = CausalLMCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    model = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=512,
        module_names=["core", "coding", "debugging", "math", "planning"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            active_modules=batch["target_modules"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        if step == 1:
            print(f"active_modules_batch0: {outputs['active_modules'][0]}")
        print(f"step {step}: loss={loss.item():.4f}")
        if step >= 3:
            break

    print("TinyMoP CPU smoke run complete.")


if __name__ == "__main__":
    main()
