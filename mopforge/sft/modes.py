"""First-class FT/SFT training mode metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_TRAINING_MODES = (
    "sft_full",
    "sft_module",
    "sft_adapter",
    "sft_generated",
    "sft_router",
    "repair_sft",
    "continued_pretraining_smoke",
)


@dataclass(slots=True)
class TrainingModeSpec:
    """Static metadata describing one fine-tuning mode."""

    mode: str
    description: str
    objective: str
    expected_policy_mode: str
    expected_model_type: str | None = None
    requires_target_modules: bool = False
    requires_fast_adapters: bool = False
    requires_generated_params: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mode spec."""

        return {
            "mode": self.mode,
            "description": self.description,
            "objective": self.objective,
            "expected_policy_mode": self.expected_policy_mode,
            "expected_model_type": self.expected_model_type,
            "requires_target_modules": self.requires_target_modules,
            "requires_fast_adapters": self.requires_fast_adapters,
            "requires_generated_params": self.requires_generated_params,
            "metadata": dict(self.metadata),
        }


_MODE_SPECS = {
    "sft_full": TrainingModeSpec(
        mode="sft_full",
        description="Supervised full-model fine-tuning over KnowledgeLesson input/output pairs.",
        objective="supervised input -> expected_output",
        expected_policy_mode="all",
    ),
    "sft_module": TrainingModeSpec(
        mode="sft_module",
        description="Supervised tuning of selected TinyMoP module parameters.",
        objective="module-targeted supervised input -> expected_output",
        expected_policy_mode="target_modules_only",
        expected_model_type="mop_oracle",
        requires_target_modules=True,
    ),
    "sft_adapter": TrainingModeSpec(
        mode="sft_adapter",
        description="Supervised adapter-only tuning with the base tiny model frozen.",
        objective="adapter-only supervised input -> expected_output",
        expected_policy_mode="fast_adapters_only",
        expected_model_type="mop_oracle",
        requires_fast_adapters=True,
    ),
    "sft_router": TrainingModeSpec(
        mode="sft_router",
        description="Supervised router smoke training from task text to target modules.",
        objective="task text -> target module mask",
        expected_policy_mode="router_only",
        expected_model_type="mop_learned_router",
        requires_target_modules=True,
        metadata={
            "mvp_limitation": (
                "Uses the existing TinyTrainer learned-router smoke path; this "
                "is not a dedicated production router trainer."
            )
        },
    ),
    "sft_generated": TrainingModeSpec(
        mode="sft_generated",
        description="Supervised generated-parameter tuning with the base tiny model frozen.",
        objective="generated-parameter supervised input -> expected_output",
        expected_policy_mode="generated_params_only",
        expected_model_type="mop_oracle",
        requires_target_modules=True,
        requires_generated_params=True,
    ),
    "repair_sft": TrainingModeSpec(
        mode="repair_sft",
        description="Supervised tuning over repair-oriented lessons.",
        objective="repair prompt -> verified target output",
        expected_policy_mode="all",
        metadata={
            "default_skill_filter": "repair",
            "default_curriculum_strategy": "repair_boosted",
        },
    ),
    "continued_pretraining_smoke": TrainingModeSpec(
        mode="continued_pretraining_smoke",
        description="Tiny causal-LM continuation smoke mode, not real pretraining.",
        objective="causal-LM smoke continuation over lesson text",
        expected_policy_mode="all",
        metadata={"not_large_scale_pretraining": True},
    ),
}


def list_training_modes() -> list[str]:
    """Return supported FT/SFT mode names in stable order."""

    return list(SUPPORTED_TRAINING_MODES)


def get_training_mode_spec(mode: str) -> TrainingModeSpec:
    """Return the static spec for ``mode``."""

    try:
        return _MODE_SPECS[mode]
    except KeyError as exc:
        valid = ", ".join(SUPPORTED_TRAINING_MODES)
        raise ValueError(f"Unsupported training mode {mode!r}. Valid modes: {valid}.") from exc
