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


def test_sparse_memory_estimate_scopes_gradients_and_adam_state_to_trainable_params():
    dense = estimate_training_memory(1_000_000, "bf16")
    sparse = estimate_training_memory(
        1_000_000,
        "bf16",
        trainable_parameter_count=10_000,
    )

    assert sparse.weight_memory_gb == dense.weight_memory_gb
    assert sparse.gradient_memory_gb < dense.gradient_memory_gb
    assert sparse.optimizer_memory_gb < dense.optimizer_memory_gb


def test_fsdp_shard_factor_reduces_persistent_state_not_activations():
    unsharded = estimate_training_memory(
        2_000_000,
        "bf16",
        d_model=128,
        n_layers=4,
        distributed_shard_factor=1,
    )
    sharded = estimate_training_memory(
        2_000_000,
        "bf16",
        d_model=128,
        n_layers=4,
        distributed_shard_factor=8,
    )

    assert sharded.weight_memory_gb < unsharded.weight_memory_gb
    assert sharded.optimizer_memory_gb < unsharded.optimizer_memory_gb
    assert sharded.activation_memory_gb_estimate == unsharded.activation_memory_gb_estimate
    assert sharded.assumptions["distributed_shard_factor"] == 8
