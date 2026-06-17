from mopforge.gpu import estimate_training_memory


def test_memory_estimator_positive_and_checkpointing_reduces_activation_estimate():
    eager = estimate_training_memory(100_000, "bf16", d_model=64, n_layers=2, activation_checkpointing=False)
    checked = estimate_training_memory(100_000, "bf16", d_model=64, n_layers=2, activation_checkpointing=True)
    assert eager.total_memory_gb_estimate > 0
    assert checked.activation_memory_gb_estimate < eager.activation_memory_gb_estimate


def test_memory_estimator_warns_when_exceeding_target():
    estimate = estimate_training_memory(1_000_000_000, "fp32", gpu_memory_gb=1)
    assert estimate.fits is False
    assert estimate.warnings
