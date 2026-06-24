"""Production post-training APIs."""

from mopforge.posttrain.preference import (
    PreferenceRecord,
    PreferenceTrainer,
    PreferenceTrainingConfig,
    build_verified_preference_records,
    cache_reference_log_probs,
    collate_preference_batch,
    dpo_loss,
    load_preference_records,
    orpo_loss,
    sequence_log_probs,
    write_preference_records,
)

__all__ = [
    "PreferenceRecord",
    "PreferenceTrainer",
    "PreferenceTrainingConfig",
    "build_verified_preference_records",
    "cache_reference_log_probs",
    "collate_preference_batch",
    "dpo_loss",
    "load_preference_records",
    "orpo_loss",
    "sequence_log_probs",
    "write_preference_records",
]
