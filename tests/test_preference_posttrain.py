import json

import pytest

from mopforge.gpu import GPUTrainingConfig, GPUTrainingState, save_gpu_checkpoint
from mopforge.models import ModelArchitectureConfig, build_tiny_model_from_architecture
from mopforge.posttrain import (
    PreferenceRecord,
    PreferenceTrainer,
    PreferenceTrainingConfig,
    build_verified_preference_records,
    dpo_loss,
    load_preference_records,
    orpo_loss,
    write_preference_records,
)
from mopforge.tokenization import ByteTokenizer


torch = pytest.importorskip("torch")


def test_dpo_and_orpo_losses_prefer_positive_margin():
    better, better_margin = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-2.0]),
        torch.tensor([-3.0]),
        beta=1.0,
    )
    worse, _ = dpo_loss(
        torch.tensor([-3.0]),
        torch.tensor([-1.0]),
        torch.tensor([-2.0]),
        torch.tensor([-3.0]),
        beta=1.0,
    )
    assert better < worse
    assert better_margin.item() > 0
    loss, preference = orpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-0.5]),
        coefficient=0.1,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(preference).all()


def test_verified_generation_artifact_builds_preference_pairs(tmp_path):
    lesson = {
        "id": "repair-1",
        "domain": "coding",
        "skill": "repair",
        "subskill": "return",
        "difficulty": 1,
        "target_modules": ["coding"],
        "input": "def add(a, b):\n    pass",
        "expected_output": "def add(a, b):\n    return a + b",
        "verification": {"type": "tests", "status": "verified"},
        "metadata": {"quality_output_format": "fixed_code_xml"},
    }
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(json.dumps(lesson) + "\n", encoding="utf-8")
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lesson_id": "repair-1",
                        "generated_text": "<fixed_code>pass</fixed_code>",
                        "passed": False,
                        "failure_type": "assertion_error",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = build_verified_preference_records(lessons, generation)
    path = write_preference_records(records, tmp_path / "preferences.jsonl")

    loaded = load_preference_records(path)
    assert len(loaded) == 1
    assert loaded[0].record_id == "repair-1"
    assert "return a + b" in loaded[0].chosen


def test_orpo_trainer_updates_and_writes_loadable_checkpoint(tmp_path):
    tokenizer = ByteTokenizer()
    architecture = ModelArchitectureConfig(
        name="orpo-smoke",
        architecture_family="production_decoder_v2",
        model_type="dense",
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_key_value_heads=1,
        intermediate_size=32,
        max_seq_len=24,
    )
    model = build_tiny_model_from_architecture(architecture, tokenizer=tokenizer)
    source_config = GPUTrainingConfig(
        name="orpo-smoke",
        architecture_family="production_decoder_v2",
        tokenizer_type="byte",
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_key_value_heads=1,
        intermediate_size=32,
        max_seq_len=24,
        device="cpu",
        precision="fp32",
        enable_amp=False,
    )
    checkpoint = tmp_path / "source.pt"
    save_gpu_checkpoint(
        checkpoint,
        model=model,
        state=GPUTrainingState(),
        config=source_config,
        model_metadata={"architecture": architecture.to_dict()},
    )
    preference_path = write_preference_records(
        [
            PreferenceRecord(
                prompt="fix: ",
                chosen="return 1",
                rejected="pass",
            )
        ],
        tmp_path / "preferences.jsonl",
    )
    result = PreferenceTrainer(
        PreferenceTrainingConfig(
            checkpoint_path=str(checkpoint),
            preference_path=str(preference_path),
            output_dir=str(tmp_path / "run"),
            objective="orpo",
            learning_rate=1e-4,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            max_seq_len=24,
            device="cpu",
            precision="fp32",
        )
    ).train()

    assert result["optimizer_steps"] == 1
    assert result["microsteps"] == 1
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert payload["model_metadata"]["posttraining"]["objective"] == "orpo"
