"""CPU smoke test: tiny learned router feeding TinyMoP active modules."""

from __future__ import annotations

from pathlib import Path

from mopforge.builders import generate_coding_bugfix_lessons
from mopforge.data import CausalLMCollator, LessonCausalLMDataset, RouterCollator, RouterDataset
from mopforge.kts import LessonStore
from mopforge.models import TinyMoPCausalTransformer, TinyModuleRouter
from mopforge.tokenization import ByteTokenizer
from mopforge.training import DEFAULT_KNOWN_MODULES, route_batch_with_router


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_bugfix_lessons.jsonl"


def ensure_lessons() -> None:
    """Create the generated coding/debugging dataset if it is missing."""

    if DATA_PATH.exists():
        return
    lessons = generate_coding_bugfix_lessons(count_per_category=10, verify=True)
    LessonStore(DATA_PATH).add_many(lesson for lesson in lessons if lesson.is_verified)


def main() -> None:
    """Train a tiny router briefly, then use it for one TinyMoP step."""

    if (
        RouterCollator is None
        or CausalLMCollator is None
        or TinyModuleRouter is None
        or TinyMoPCausalTransformer is None
    ):
        print("PyTorch is not installed; skipping learned-router TinyMoP smoke test.")
        return

    import torch
    from torch.utils.data import DataLoader

    print("TinyMoP with learned router CPU smoke test only.")
    ensure_lessons()
    torch.manual_seed(23)

    lessons = LessonStore(DATA_PATH).load_all()
    tokenizer = ByteTokenizer()

    router_dataset = RouterDataset(
        lessons,
        tokenizer,
        known_modules=DEFAULT_KNOWN_MODULES,
        max_length=512,
    )
    router_loader = DataLoader(
        router_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=RouterCollator(tokenizer),
    )
    router = TinyModuleRouter(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        hidden_dim=128,
        known_modules=DEFAULT_KNOWN_MODULES,
        pad_token_id=tokenizer.pad_token_id,
    )
    router_optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)

    router.train()
    for step, batch in enumerate(router_loader, start=1):
        router_optimizer.zero_grad(set_to_none=True)
        loss = router(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            module_mask=batch["module_mask"],
        )["loss"]
        loss.backward()
        router_optimizer.step()
        print(f"router_step={step} loss={loss.item():.4f}")
        if step >= 2:
            break

    lm_dataset = LessonCausalLMDataset(lessons, tokenizer, max_length=512)
    lm_loader = DataLoader(
        lm_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer),
    )
    mop = TinyMoPCausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=512,
        module_names=DEFAULT_KNOWN_MODULES,
    )
    mop_optimizer = torch.optim.AdamW(mop.parameters(), lr=1e-3)

    router.eval()
    mop.train()
    router_batch = next(iter(router_loader))
    lm_batch = next(iter(lm_loader))
    predicted_modules = route_batch_with_router(
        router,
        router_batch,
        DEFAULT_KNOWN_MODULES,
        threshold=0.5,
    )
    mop_optimizer.zero_grad(set_to_none=True)
    outputs = mop(
        input_ids=lm_batch["input_ids"],
        attention_mask=lm_batch["attention_mask"],
        labels=lm_batch["labels"],
        active_modules=predicted_modules,
    )
    outputs["loss"].backward()
    mop_optimizer.step()

    print(f"predicted_modules_batch0: {predicted_modules[0]}")
    print(f"mop_loss={outputs['loss'].item():.4f}")
    print("Learned-router TinyMoP CPU smoke run complete.")


if __name__ == "__main__":
    main()
