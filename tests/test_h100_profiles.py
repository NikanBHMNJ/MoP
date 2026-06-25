import json
from pathlib import Path

from mopforge.gpu import GPUTrainingConfig
from mopforge.models import ProductionDecoderConfig, production_parameter_count
from mopforge.posttrain import PreferenceTrainingConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_profile(name):
    path = ROOT / "configs" / "jobs" / name
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GPUTrainingConfig.from_dict(raw["payload"]), raw


def _exact_count(config):
    return production_parameter_count(
        ProductionDecoderConfig(
            vocab_size=32768,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_key_value_heads=config.n_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_seq_len=config.max_seq_len,
            model_type=config.model_type,
            module_names=config.module_names or ["core", "coding", "debugging", "repair"],
            active_experts=config.active_experts,
            routing_granularity=config.routing_granularity,
        )
    )


def test_h100_single_gpu_profiles_have_exact_parameter_counts_and_tiers():
    expectations = {
        "h100_300m_dense_probe.json": 304137216,
        "h100_1b_dense_probe.json": 1015779072,
        "h100_2b_dense_80gb_probe.json": 2082246912,
        "h100_2b_dense_94gb_probe.json": 2082246912,
        "h100_2b_mop_94gb_probe.json": 2480265984,
    }
    for name, expected in expectations.items():
        config, _ = _load_profile(name)
        assert config.architecture_family == "production_decoder_v2"
        assert config.metadata["parameter_count"] == expected
        assert _exact_count(config) == expected
        assert config.precision == "bf16"
        assert config.max_seq_len == 1024
        assert config.micro_batch_size == 1
        assert config.metadata["required_gpu_name_contains"] == "H100"
        assert "quantization" not in config.to_dict()


def test_h100_fsdp_pilots_use_sharded_resume_and_token_budget():
    for name in (
        "h100_2b_dense_fsdp_pilot.json",
        "h100_2b_mop_fsdp_pilot.json",
    ):
        config, envelope = _load_profile(name)
        assert config.distributed_strategy == "fsdp"
        assert config.distributed_checkpoint_mode == "sharded"
        assert config.max_optimizer_steps == 500
        assert config.scheduler_unit == "tokens"
        assert config.max_train_tokens == 16777216
        assert config.metadata["distributed_world_size"] == 8
        assert envelope["metadata"]["distributed"]["nproc_per_node"] == 8


def test_h100_notebook_enforces_ordered_tiers_and_report_boundary():
    path = ROOT / "notebooks" / "colab_h100_2b_readiness.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    assert "'H100' not in props.name.upper()" in source
    assert "75 <= total_gb <= 86" in source
    assert "86 < total_gb <= 100" in source
    assert source.index("300M gate passed") < source.index("1B gate passed")
    assert source.index("1B gate passed") < source.index("2B admission gate passed")
    assert "contains_weights': False" in source


def test_h100_preference_configs_parse_as_dpo_and_orpo():
    dpo = PreferenceTrainingConfig.from_json(
        ROOT / "configs" / "posttrain" / "h100_2b_dpo.json"
    )
    orpo = PreferenceTrainingConfig.from_json(
        ROOT / "configs" / "posttrain" / "h100_2b_orpo.json"
    )
    assert dpo.objective == "dpo"
    assert dpo.reference_checkpoint_path == dpo.checkpoint_path
    assert orpo.objective == "orpo"
    assert orpo.reference_checkpoint_path is None
