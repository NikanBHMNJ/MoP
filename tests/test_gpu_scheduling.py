import pytest

from mopforge.gpu.scheduling import TokenLRScheduler


torch = pytest.importorskip("torch")


def test_token_lr_scheduler_warmup_cosine_and_resume():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = TokenLRScheduler(
        optimizer,
        scheduler="cosine",
        total_tokens=1000,
        warmup_tokens=100,
        min_lr_ratio=0.1,
    )

    scheduler.step(50)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    scheduler.step(100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    scheduler.step(1000)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    restored = TokenLRScheduler(
        optimizer,
        scheduler="cosine",
        total_tokens=1000,
        warmup_tokens=100,
        min_lr_ratio=0.1,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.tokens_seen == 1000
    assert restored.get_last_lr()[0] == pytest.approx(0.1)
