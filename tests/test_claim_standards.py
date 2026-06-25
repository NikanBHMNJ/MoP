from __future__ import annotations

import json
from pathlib import Path

from mopforge.claims import load_claim_card, scaffold_claim_card, validate_claim_card, write_claim_card
from mopforge.cli.main import main


ROOT = Path(__file__).resolve().parents[1]


def test_claim_standard_docs_are_linked() -> None:
    expected_paths = [
        ROOT / "docs" / "academic_claim_standard.md",
        ROOT / "docs" / "startup_product_claim_standard.md",
        ROOT / "reports" / "claim_readiness_template" / "README.md",
        ROOT / "reports" / "claim_readiness_template" / "claim_readiness_schema.json",
    ]
    for path in expected_paths:
        assert path.exists(), f"missing claim standard artifact: {path}"

    linked_docs = {
        ROOT / "README.md": [
            "docs/academic_claim_standard.md",
            "docs/startup_product_claim_standard.md",
            "reports/claim_readiness_template/",
        ],
        ROOT / "docs" / "README.md": [
            "academic_claim_standard.md",
            "startup_product_claim_standard.md",
            "../reports/claim_readiness_template/README.md",
        ],
        ROOT / "AGENTS.md": [
            "docs/academic_claim_standard.md",
            "docs/startup_product_claim_standard.md",
            "reports/claim_readiness_template/",
        ],
    }
    for path, links in linked_docs.items():
        text = path.read_text(encoding="utf-8")
        for link in links:
            assert link in text, f"{path} does not link {link}"


def test_claim_readiness_schema_levels() -> None:
    schema_path = ROOT / "reports" / "claim_readiness_template" / "claim_readiness_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert "academic_level" in schema["required"]
    assert schema["properties"]["academic_level"]["enum"] == ["A0", "A1", "A2", "A3", "A4", "A5"]
    assert schema["properties"]["product_level"]["enum"] == [
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "not_applicable",
    ]
    assert set(schema["properties"]["decision"]["enum"]) == {
        "publish",
        "publish-narrowly",
        "rerun",
        "block",
    }


def test_verified_code_repair_claim_card_passes_a2_p2(tmp_path: Path) -> None:
    card = scaffold_claim_card(
        report_dir=ROOT / "reports" / "verified_code_repair_100m_l4",
        claim_statement=(
            "MoP-Forge measures narrow verified 100M code-repair efficiency on the fixed L4 report split."
        ),
        academic_level="A2",
        product_level="P2",
        root=ROOT,
    )
    validation = validate_claim_card(card, root=ROOT)

    assert validation["overall_passed"], validation
    assert validation["decision_recommendation"] == "publish-narrowly"
    assert card.metrics["acceptance_gate_passed"] is True
    assert card.metrics["best_cached_peak_reserved_reduction_vs_dense_x"] > 3.0

    path = write_claim_card(card, tmp_path / "claim_card.json")
    assert load_claim_card(path).claim_id == "verified_code_repair_100m_l4"


def test_external_or_frontier_claims_are_blocked_without_evidence() -> None:
    card = scaffold_claim_card(
        report_dir=ROOT / "reports" / "verified_code_repair_100m_l4",
        claim_statement="MoP-Forge is a Qwen-class frontier code model.",
        academic_level="A4",
        product_level="P3",
        root=ROOT,
    )
    validation = validate_claim_card(card, root=ROOT)

    assert not validation["overall_passed"]
    assert "a4_external_benchmark_and_contamination" in validation["failed_required_gates"]
    assert "high_model_quality_wording_supported" in validation["failed_required_gates"]


def test_claim_cli_scaffold_and_validate(tmp_path: Path, capsys) -> None:
    card_path = tmp_path / "verified_code_repair_claim_card.json"
    assert (
        main(
            [
                "claim",
                "scaffold",
                "--report-dir",
                "reports/verified_code_repair_100m_l4",
                "--claim-statement",
                "MoP-Forge supports a narrow verified code-repair efficiency claim.",
                "--academic-level",
                "A2",
                "--product-level",
                "P2",
                "--output",
                str(card_path),
            ]
        )
        == 0
    )
    scaffold_output = capsys.readouterr().out
    assert "claim_card_path=" in scaffold_output
    assert "overall_passed=True" in scaffold_output

    assert main(["claim", "validate", str(card_path), "--root", str(ROOT)]) == 0
    validate_output = capsys.readouterr().out
    assert "decision_recommendation=publish-narrowly" in validate_output
