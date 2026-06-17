"""Train the tiny learned router for a few CPU smoke-test steps."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import RouterCollator, RouterDataset
from mopforge.kts import LessonStore
from mopforge.models import TinyModuleRouter, predict_modules
from mopforge.tokenization import ByteTokenizer
from mopforge.training import DEFAULT_KNOWN_MODULES


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if DATA_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(DATA_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Run three tiny CPU-only router optimization steps."""

    if RouterCollator is None or TinyModuleRouter is None:
        print("PyTorch is not installed; skipping tiny router CPU smoke test.")
        return

    import torch
    from torch.utils.data import DataLoader

    print("Tiny router CPU smoke test only. Losses are not meaningful.")
    ensure_lessons()
    torch.manual_seed(17)

    lessons = LessonStore(DATA_PATH).load_all()
    tokenizer = ByteTokenizer()
    dataset = RouterDataset(
        lessons,
        tokenizer,
        known_modules=DEFAULT_KNOWN_MODULES,
        max_length=512,
    )
    collator = RouterCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    router = TinyModuleRouter(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        hidden_dim=128,
        known_modules=DEFAULT_KNOWN_MODULES,
        pad_token_id=tokenizer.pad_token_id,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)

    last_batch = None
    router.train()
    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = router(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            module_mask=batch["module_mask"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        last_batch = batch
        print(f"step={step} loss={loss.item():.4f}")
        if step >= 3:
            break

    if last_batch is not None:
        router.eval()
        with torch.no_grad():
            outputs = router(
                input_ids=last_batch["input_ids"],
                attention_mask=last_batch["attention_mask"],
            )
        predicted = predict_modules(
            outputs["logits"][0],
            DEFAULT_KNOWN_MODULES,
            threshold=0.5,
        )
        print(f"target_modules: {last_batch['target_modules'][0]}")
        print(f"predicted_modules: {predicted}")


if __name__ == "__main__":
    main()
