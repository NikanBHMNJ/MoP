"""Academic and product claim gates for report artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import mopforge


ACADEMIC_LEVELS = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5}
PRODUCT_LEVELS = {
    "not_applicable": -1,
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
}
CLAIM_TYPES = {"academic", "product", "release", "benchmark"}
EVIDENCE_STATUSES = {"implemented", "measured", "repeated", "external", "blocked"}
DECISIONS = {"publish", "publish-narrowly", "rerun", "block"}
BLOCKED_ARTIFACT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".npz",
    ".npy",
}


@dataclass(slots=True)
class ClaimCard:
    claim_id: str
    claim_statement: str
    claim_type: str
    academic_level: str
    product_level: str
    evidence_status: str
    version: str
    commit: str
    report_dir: str
    dataset: dict[str, Any]
    hardware: dict[str, Any]
    baselines: list[str]
    metrics: dict[str, Any]
    limitations: list[str]
    allowed_wording: list[str]
    blocked_wording: list[str]
    decision: str
    evidence_files: list[str] = field(default_factory=list)
    artifact_audit: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    claim_card_format: str = "mopforge_claim_card_v1"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimCard":
        payload = dict(data)
        payload.setdefault("evidence_files", [])
        payload.setdefault("artifact_audit", {})
        payload.setdefault("created_utc", "")
        payload.setdefault("claim_card_format", "mopforge_claim_card_v1")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_card_format": self.claim_card_format,
            "claim_id": self.claim_id,
            "claim_statement": self.claim_statement,
            "claim_type": self.claim_type,
            "academic_level": self.academic_level,
            "product_level": self.product_level,
            "evidence_status": self.evidence_status,
            "version": self.version,
            "commit": self.commit,
            "report_dir": self.report_dir,
            "dataset": self.dataset,
            "hardware": self.hardware,
            "baselines": self.baselines,
            "metrics": self.metrics,
            "limitations": self.limitations,
            "allowed_wording": self.allowed_wording,
            "blocked_wording": self.blocked_wording,
            "decision": self.decision,
            "evidence_files": self.evidence_files,
            "artifact_audit": self.artifact_audit,
            "created_utc": self.created_utc,
        }


@dataclass(slots=True)
class ClaimGate:
    name: str
    required: bool
    passed: bool | None
    observed: Any = None
    threshold: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def scaffold_claim_card(
    *,
    report_dir: str | Path,
    claim_statement: str,
    claim_type: str = "benchmark",
    academic_level: str = "A2",
    product_level: str = "P1",
    version: str | None = None,
    commit: str | None = None,
    root: str | Path = ".",
) -> ClaimCard:
    """Build a first-pass claim card from a report directory."""

    root_path = Path(root).resolve()
    report_path = _resolve_report_dir(report_dir, root_path)
    report_ref = _relative_ref(report_path, root_path)
    bundle = _load_report_bundle(report_path)
    metrics = _summarize_report_metrics(bundle)
    dataset = _summarize_dataset(bundle)
    hardware = _summarize_hardware(bundle)
    baselines = _summarize_baselines(bundle)
    evidence_files = _evidence_files(report_path, root_path)
    artifact_audit = audit_report_artifacts(report_path)
    evidence_status = _infer_evidence_status(bundle, metrics)
    decision = "publish-narrowly" if evidence_status in {"measured", "repeated", "external"} else "block"
    if evidence_status == "blocked":
        decision = "block"
    return ClaimCard(
        claim_id=report_path.name,
        claim_statement=claim_statement,
        claim_type=claim_type,
        academic_level=academic_level,
        product_level=product_level,
        evidence_status=evidence_status,
        version=version or mopforge.__version__,
        commit=commit or _git_commit(root_path),
        report_dir=report_ref,
        dataset=dataset,
        hardware=hardware,
        baselines=baselines,
        metrics=metrics,
        limitations=_default_limitations(academic_level, product_level, bundle),
        allowed_wording=[claim_statement],
        blocked_wording=_default_blocked_wording(),
        decision=decision,
        evidence_files=evidence_files,
        artifact_audit=artifact_audit,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )


def validate_claim_card(card: ClaimCard | dict[str, Any], *, root: str | Path = ".") -> dict[str, Any]:
    """Validate whether a claim card supports its requested claim levels."""

    if isinstance(card, dict):
        card = ClaimCard.from_dict(card)
    root_path = Path(root).resolve()
    report_path = _resolve_report_dir(card.report_dir, root_path)
    gates = _schema_gates(card)
    gates.extend(_report_gates(card, report_path, root_path))
    gates.extend(_academic_level_gates(card, report_path))
    gates.extend(_product_level_gates(card))
    gates.extend(_wording_gates(card))
    required = [gate for gate in gates if gate.required]
    failed = [gate.name for gate in required if gate.passed is False]
    unknown = [gate.name for gate in required if gate.passed is None]
    overall_passed = not failed and not unknown
    recommendation = _decision_recommendation(card, overall_passed)
    return {
        "kind": "mopforge_claim_readiness_report_v1",
        "claim_id": card.claim_id,
        "claim_statement": card.claim_statement,
        "requested_academic_level": card.academic_level,
        "requested_product_level": card.product_level,
        "evidence_status": card.evidence_status,
        "report_dir": card.report_dir,
        "overall_passed": overall_passed,
        "decision_recommendation": recommendation,
        "failed_required_gates": failed,
        "unknown_required_gates": unknown,
        "gates": [gate.to_dict() for gate in gates],
        "allowed_wording": card.allowed_wording,
        "blocked_wording": card.blocked_wording,
    }


def load_claim_card(path: str | Path) -> ClaimCard:
    return ClaimCard.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_claim_card(card: ClaimCard, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def write_claim_validation(validation: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return output


def format_claim_validation(validation: dict[str, Any]) -> str:
    lines = [
        f"claim_id={validation.get('claim_id')}",
        f"academic_level={validation.get('requested_academic_level')}",
        f"product_level={validation.get('requested_product_level')}",
        f"evidence_status={validation.get('evidence_status')}",
        f"overall_passed={validation.get('overall_passed')}",
        f"decision_recommendation={validation.get('decision_recommendation')}",
    ]
    failed = validation.get("failed_required_gates") or []
    unknown = validation.get("unknown_required_gates") or []
    if failed:
        lines.append("failed_required_gates=" + ",".join(failed))
    if unknown:
        lines.append("unknown_required_gates=" + ",".join(unknown))
    return "\n".join(lines)


def audit_report_artifacts(report_dir: str | Path) -> dict[str, Any]:
    path = Path(report_dir)
    blocked: list[str] = []
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in BLOCKED_ARTIFACT_SUFFIXES:
                blocked.append(str(item))
    return {
        "checked": path.exists(),
        "blocked_suffixes": sorted(BLOCKED_ARTIFACT_SUFFIXES),
        "blocked_artifacts": blocked,
        "weights_allowed": False,
        "passed": not blocked if path.exists() else None,
    }


def _schema_gates(card: ClaimCard) -> list[ClaimGate]:
    required_fields = [
        "claim_id",
        "claim_statement",
        "claim_type",
        "academic_level",
        "product_level",
        "evidence_status",
        "version",
        "commit",
        "report_dir",
        "dataset",
        "hardware",
        "baselines",
        "metrics",
        "limitations",
        "allowed_wording",
        "blocked_wording",
        "decision",
    ]
    missing = [name for name in required_fields if _empty(getattr(card, name))]
    enum_errors = []
    if card.claim_type not in CLAIM_TYPES:
        enum_errors.append("claim_type")
    if card.academic_level not in ACADEMIC_LEVELS:
        enum_errors.append("academic_level")
    if card.product_level not in PRODUCT_LEVELS:
        enum_errors.append("product_level")
    if card.evidence_status not in EVIDENCE_STATUSES:
        enum_errors.append("evidence_status")
    if card.decision not in DECISIONS:
        enum_errors.append("decision")
    return [
        ClaimGate(
            "claim_card_required_fields",
            True,
            not missing,
            observed=missing,
            threshold="all required fields present",
        ),
        ClaimGate(
            "claim_card_enum_values",
            True,
            not enum_errors,
            observed=enum_errors,
            threshold="known claim/evidence/decision levels",
        ),
    ]


def _report_gates(card: ClaimCard, report_path: Path, root_path: Path) -> list[ClaimGate]:
    report_under_reports = _relative_ref(report_path, root_path).replace("\\", "/").startswith("reports/")
    evidence_file_count = len(card.evidence_files)
    audit = card.artifact_audit or audit_report_artifacts(report_path)
    blocked_artifacts = list(audit.get("blocked_artifacts") or [])
    gates = [
        ClaimGate(
            "report_dir_under_reports",
            True,
            report_under_reports,
            observed=_relative_ref(report_path, root_path),
            threshold="reports/<claim-id>",
        ),
        ClaimGate(
            "report_dir_exists",
            True,
            report_path.exists() and report_path.is_dir(),
            observed=str(report_path),
            threshold=True,
        ),
        ClaimGate(
            "lightweight_artifact_audit",
            True,
            not blocked_artifacts,
            observed=blocked_artifacts,
            threshold="no model weights, checkpoints, caches, arrays, or token shards in report",
        ),
        ClaimGate(
            "evidence_files_present",
            _academic_rank(card.academic_level) >= 1 or _product_rank(card.product_level) >= 1,
            evidence_file_count > 0,
            observed=evidence_file_count,
            threshold="at least one report artifact",
        ),
    ]
    return gates


def _academic_level_gates(card: ClaimCard, report_path: Path) -> list[ClaimGate]:
    rank = _academic_rank(card.academic_level)
    metrics = card.metrics
    dataset = card.dataset
    controlled_fields = {
        "run_count": metrics.get("run_count"),
        "baselines": len(card.baselines),
        "dataset_name": dataset.get("name"),
        "split_id": dataset.get("split_id"),
        "split_seed": dataset.get("split_seed"),
    }
    repeated_count = _coerce_int(metrics.get("seed_count") or metrics.get("repeated_run_count") or metrics.get("run_seed_count"))
    external_ready = bool(metrics.get("contamination_audit_passed")) and bool(
        metrics.get("external_benchmark_count") or metrics.get("pass_at_1") is not None
    )
    gates = [
        ClaimGate(
            "a0_implemented_mechanism",
            rank >= 0,
            bool(card.version and card.commit),
            observed={"version": card.version, "commit": card.commit},
            threshold="version and commit recorded",
        ),
        ClaimGate(
            "a1_single_run_evidence",
            rank >= 1,
            bool(metrics.get("has_comparison_json") or metrics.get("has_acceptance_gates_json") or metrics.get("has_report_schema_json")),
            observed={key: metrics.get(key) for key in ("has_comparison_json", "has_acceptance_gates_json", "has_report_schema_json")},
            threshold="report schema, comparison, or acceptance gate artifact",
        ),
        ClaimGate(
            "a2_controlled_comparison",
            rank >= 2,
            _controlled_comparison_ready(controlled_fields),
            observed=controlled_fields,
            threshold="same-split comparison with at least two runs and a baseline",
        ),
        ClaimGate(
            "a3_repeated_comparison",
            rank >= 3,
            repeated_count >= 3 or card.evidence_status in {"repeated", "external"},
            observed={"repeated_count": repeated_count, "evidence_status": card.evidence_status},
            threshold="at least three seeds/repeats or repeated evidence status",
        ),
        ClaimGate(
            "a4_external_benchmark_and_contamination",
            rank >= 4,
            external_ready or card.evidence_status == "external",
            observed={
                "contamination_audit_passed": metrics.get("contamination_audit_passed"),
                "external_benchmark_count": metrics.get("external_benchmark_count"),
                "pass_at_1": metrics.get("pass_at_1"),
                "evidence_status": card.evidence_status,
            },
            threshold="external benchmark plus contamination evidence",
        ),
        ClaimGate(
            "a5_paper_ready_bundle",
            rank >= 5,
            (external_ready or card.evidence_status == "external") and bool(card.limitations) and report_path.exists(),
            observed={"limitations": len(card.limitations), "report_exists": report_path.exists()},
            threshold="A4 evidence, limitations, and reproducible report bundle",
        ),
    ]
    return gates


def _product_level_gates(card: ClaimCard) -> list[ClaimGate]:
    rank = _product_rank(card.product_level)
    metrics = card.metrics
    hardware = card.hardware
    p2_metrics = {
        "tokens_per_sec": metrics.get("best_tokens_per_sec") or metrics.get("tokens_per_sec"),
        "throughput_ratio": metrics.get("best_cached_throughput_vs_dense_x"),
        "peak_reserved_reduction": metrics.get("best_cached_peak_reserved_reduction_vs_dense_x"),
        "checkpoint_reduction": metrics.get("best_cached_checkpoint_reduction_vs_dense_x"),
        "acceptance_gate_passed": metrics.get("acceptance_gate_passed"),
    }
    gates = [
        ClaimGate(
            "p0_research_prototype",
            rank >= 0,
            bool(card.version and card.claim_statement),
            observed={"version": card.version, "claim_statement": bool(card.claim_statement)},
            threshold="versioned claim statement",
        ),
        ClaimGate(
            "p1_reproducible_workflow",
            rank >= 1,
            bool(card.evidence_files and card.allowed_wording and card.blocked_wording),
            observed={
                "evidence_files": len(card.evidence_files),
                "allowed_wording": len(card.allowed_wording),
                "blocked_wording": len(card.blocked_wording),
            },
            threshold="report artifacts plus allowed/blocked wording",
        ),
        ClaimGate(
            "p2_pilot_ready_workflow",
            rank >= 2,
            bool(hardware.get("device") and any(value is not None for value in p2_metrics.values()) and card.limitations),
            observed={"hardware": hardware, "metrics": p2_metrics, "limitations": len(card.limitations)},
            threshold="hardware target, cost/quality/efficiency metrics, and limitations",
        ),
        ClaimGate(
            "p3_product_beta_reliability",
            rank >= 3,
            bool(metrics.get("resume_passed") and metrics.get("monitoring_ready") and metrics.get("support_process_ready")),
            observed={
                "resume_passed": metrics.get("resume_passed"),
                "monitoring_ready": metrics.get("monitoring_ready"),
                "support_process_ready": metrics.get("support_process_ready"),
            },
            threshold="resume, monitoring, and support evidence",
        ),
        ClaimGate(
            "p4_customer_proven_value",
            rank >= 4,
            _coerce_int(metrics.get("external_pilot_count") or metrics.get("customer_count")) > 0,
            observed={
                "external_pilot_count": metrics.get("external_pilot_count"),
                "customer_count": metrics.get("customer_count"),
            },
            threshold="at least one external pilot/customer result",
        ),
    ]
    return gates


def _wording_gates(card: ClaimCard) -> list[ClaimGate]:
    public_text = " ".join([card.claim_statement, *card.allowed_wording]).lower()
    rank_a = _academic_rank(card.academic_level)
    rank_p = _product_rank(card.product_level)
    metrics = card.metrics
    high_model_terms = any(term in public_text for term in ("qwen", "frontier", "usable 2b", "general code"))
    production_terms = any(term in public_text for term in ("production service", "managed production", "production-grade"))
    same_quality_terms = any(term in public_text for term in ("same quality", "beats dense", "better than dense", "superior"))
    efficiency_multiplier_terms = any(term in public_text for term in ("3x", "50x", "30x", "lower gpu", "lower vram"))
    return [
        ClaimGate(
            "high_model_quality_wording_supported",
            high_model_terms,
            (rank_a >= 4 or rank_p >= 3) and bool(metrics.get("external_benchmark_count") or metrics.get("pass_at_1") is not None),
            observed={"terms_present": high_model_terms, "academic_rank": rank_a, "product_rank": rank_p},
            threshold="A4/P3 plus external benchmark evidence",
        ),
        ClaimGate(
            "production_service_wording_supported",
            production_terms,
            rank_p >= 3 and bool(metrics.get("monitoring_ready") and metrics.get("support_process_ready")),
            observed={"terms_present": production_terms, "product_rank": rank_p},
            threshold="P3 product beta evidence",
        ),
        ClaimGate(
            "same_quality_wording_supported",
            same_quality_terms,
            rank_a >= 2 and bool(metrics.get("acceptance_gate_passed") or metrics.get("same_quality_gate_passed")),
            observed={
                "terms_present": same_quality_terms,
                "academic_rank": rank_a,
                "acceptance_gate_passed": metrics.get("acceptance_gate_passed"),
                "same_quality_gate_passed": metrics.get("same_quality_gate_passed"),
            },
            threshold="A2 controlled comparison plus quality gate",
        ),
        ClaimGate(
            "efficiency_multiplier_wording_supported",
            efficiency_multiplier_terms,
            any(
                _coerce_float(metrics.get(name)) and _coerce_float(metrics.get(name)) >= 3.0
                for name in (
                    "best_cached_peak_reserved_reduction_vs_dense_x",
                    "best_cached_peak_allocated_reduction_vs_dense_x",
                    "best_cached_checkpoint_reduction_vs_dense_x",
                    "best_cached_throughput_vs_dense_x",
                )
            ),
            observed={
                "terms_present": efficiency_multiplier_terms,
                "peak_reserved_reduction": metrics.get("best_cached_peak_reserved_reduction_vs_dense_x"),
                "checkpoint_reduction": metrics.get("best_cached_checkpoint_reduction_vs_dense_x"),
                "throughput_ratio": metrics.get("best_cached_throughput_vs_dense_x"),
            },
            threshold="named measured multiplier at or above 3x",
        ),
    ]


def _decision_recommendation(card: ClaimCard, overall_passed: bool) -> str:
    if not overall_passed:
        return "block"
    if _academic_rank(card.academic_level) >= 3 or _product_rank(card.product_level) >= 2:
        return "publish-narrowly"
    return card.decision if card.decision in {"publish", "publish-narrowly"} else "publish-narrowly"


def _load_report_bundle(report_path: Path) -> dict[str, Any]:
    names = [
        "comparison.json",
        "acceptance_gates.json",
        "run_manifest.json",
        "experiment_settings.json",
        "report_schema.json",
        "report_manifest.json",
        "readiness_summary.json",
    ]
    return {name: _read_json(report_path / name) for name in names if (report_path / name).exists()}


def _summarize_report_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    comparison = bundle.get("comparison.json") or {}
    acceptance = bundle.get("acceptance_gates.json") or {}
    schema = bundle.get("report_schema.json") or {}
    readiness = bundle.get("readiness_summary.json") or {}
    runs = list(comparison.get("runs") or [])
    dense = _first_run_matching(runs, "dense")
    cached_runs = [run for run in runs if "cached" in str(run.get("run_id", "")).lower()]
    best_cached = _best_run(cached_runs)
    metrics: dict[str, Any] = {
        "has_comparison_json": bool(comparison),
        "has_acceptance_gates_json": bool(acceptance),
        "has_report_schema_json": bool(schema),
        "has_readiness_summary_json": bool(readiness),
        "run_count": len(runs),
        "baseline_count": len([run for run in runs if _is_baseline_run(run)]),
        "acceptance_gate_passed": _acceptance_passed(acceptance),
        "acceptance_check_count": len(dict(acceptance.get("checks") or {})),
        "report_schema_status": schema.get("status"),
        "weights_allowed": schema.get("weights_allowed"),
    }
    if dense:
        metrics.update(
            {
                "dense_run_id": dense.get("run_id"),
                "dense_best_eval_loss": dense.get("best_eval_loss"),
                "dense_eval_loss": dense.get("eval_loss"),
                "dense_verifier_pass_rate": dense.get("gen_verifier_pass_rate"),
                "dense_exact_match_rate": dense.get("gen_exact_match_rate"),
                "dense_tokens_per_sec": dense.get("tokens_per_sec"),
                "dense_peak_reserved_gb": dense.get("peak_reserved_gb"),
                "dense_peak_allocated_gb": dense.get("peak_allocated_gb"),
                "dense_checkpoint_size_mb": dense.get("checkpoint_size_mb"),
            }
        )
    if best_cached:
        metrics.update(
            {
                "best_cached_run_id": best_cached.get("run_id"),
                "best_cached_best_eval_loss": best_cached.get("best_eval_loss"),
                "best_cached_eval_loss": best_cached.get("eval_loss"),
                "best_cached_verifier_pass_rate": best_cached.get("gen_verifier_pass_rate"),
                "best_cached_exact_match_rate": best_cached.get("gen_exact_match_rate"),
                "best_cached_tokens_per_sec": best_cached.get("tokens_per_sec"),
                "best_cached_peak_reserved_gb": best_cached.get("peak_reserved_gb"),
                "best_cached_peak_allocated_gb": best_cached.get("peak_allocated_gb"),
                "best_cached_checkpoint_size_mb": best_cached.get("checkpoint_size_mb"),
            }
        )
    if dense and best_cached:
        metrics.update(
            {
                "best_cached_throughput_vs_dense_x": _ratio(best_cached.get("tokens_per_sec"), dense.get("tokens_per_sec")),
                "best_cached_peak_reserved_reduction_vs_dense_x": _ratio(dense.get("peak_reserved_gb"), best_cached.get("peak_reserved_gb")),
                "best_cached_peak_allocated_reduction_vs_dense_x": _ratio(dense.get("peak_allocated_gb"), best_cached.get("peak_allocated_gb")),
                "best_cached_checkpoint_reduction_vs_dense_x": _ratio(dense.get("checkpoint_size_mb"), best_cached.get("checkpoint_size_mb")),
                "best_cached_verifier_delta_vs_dense": _delta(best_cached.get("gen_verifier_pass_rate"), dense.get("gen_verifier_pass_rate")),
                "best_cached_exact_match_delta_vs_dense": _delta(best_cached.get("gen_exact_match_rate"), dense.get("gen_exact_match_rate")),
            }
        )
    evidence = dict(acceptance.get("evidence") or {})
    if evidence:
        metrics["profile_evidence_count"] = len(evidence)
        metrics["max_full_eval_examples"] = max((_coerce_int(dict(item).get("full_eval_examples")) for item in evidence.values()), default=0)
        metrics["max_optimizer_updates"] = max((_coerce_int(dict(item).get("optimizer_updates")) for item in evidence.values()), default=0)
    if readiness:
        metrics.update(
            {
                "readiness_probe_completed": readiness.get("probe_completed"),
                "checkpoint_resume_passed": readiness.get("checkpoint_resume_passed"),
                "resume_passed": readiness.get("checkpoint_resume_passed"),
                "peak_reserved_within_limit": readiness.get("peak_reserved_within_limit"),
                "no_oom": readiness.get("no_oom"),
            }
        )
    return {key: value for key, value in metrics.items() if value is not None}


def _summarize_dataset(bundle: dict[str, Any]) -> dict[str, Any]:
    settings = bundle.get("experiment_settings.json") or {}
    return {
        "name": settings.get("dataset_ref") or settings.get("dataset") or settings.get("report_id") or "unknown",
        "split_id": settings.get("dataset_split_id") or settings.get("split_id") or "unknown",
        "split_seed": settings.get("split_seed", "unknown"),
        "stratify_by": settings.get("split_stratify_by") or settings.get("generation_eval_stratify_by"),
        "quality_format": settings.get("quality_format"),
    }


def _summarize_hardware(bundle: dict[str, Any]) -> dict[str, Any]:
    settings = bundle.get("experiment_settings.json") or {}
    runs = list((bundle.get("comparison.json") or {}).get("runs") or [])
    first_run = runs[0] if runs else {}
    return {
        "device": settings.get("hardware_target") or first_run.get("selected_device") or "unknown",
        "memory_gb": settings.get("hardware_memory_gb") or "unknown",
        "precision": first_run.get("selected_precision") or settings.get("precision"),
    }


def _summarize_baselines(bundle: dict[str, Any]) -> list[str]:
    runs = list((bundle.get("comparison.json") or {}).get("runs") or [])
    baselines = [str(run.get("run_id")) for run in runs if _is_baseline_run(run)]
    return [item for item in baselines if item and item != "None"]


def _infer_evidence_status(bundle: dict[str, Any], metrics: dict[str, Any]) -> str:
    schema = bundle.get("report_schema.json") or {}
    if schema.get("status") and "awaiting" in str(schema.get("status")).lower():
        return "implemented"
    passed = metrics.get("acceptance_gate_passed")
    if passed is False:
        return "blocked"
    if passed is True or metrics.get("has_comparison_json"):
        return "measured"
    return "implemented"


def _evidence_files(report_path: Path, root_path: Path) -> list[str]:
    if not report_path.exists():
        return []
    allowed_suffixes = {".json", ".csv", ".md", ".txt"}
    files = [
        _relative_ref(path, root_path)
        for path in sorted(report_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]
    return files


def _default_limitations(
    academic_level: str,
    product_level: str,
    bundle: dict[str, Any],
) -> list[str]:
    limitations = [
        "Claim is scoped to the named report directory, dataset split, seed, hardware target, and training budget.",
        "Large checkpoints, caches, token shards, optimizer states, and model weights are not part of the lightweight evidence bundle.",
    ]
    if _academic_rank(academic_level) >= 3:
        limitations.append("Repeated-seed evidence is required before using this as a strong comparative research claim.")
    if _academic_rank(academic_level) >= 4:
        limitations.append("External benchmark and contamination evidence are required before comparing to public models.")
    if _product_rank(product_level) >= 2:
        limitations.append("Pilot readiness remains scoped to the measured workflow and hardware target.")
    if (bundle.get("report_schema.json") or {}).get("status"):
        limitations.append(f"Report schema status: {(bundle.get('report_schema.json') or {}).get('status')}.")
    return limitations


def _default_blocked_wording() -> list[str]:
    return [
        "Qwen-class, frontier-class, or generally useful code-model quality without external benchmark evidence.",
        "Production-grade managed training service without reliability, security, monitoring, recovery, and support evidence.",
        "Guaranteed 1B/2B training on unseen hardware.",
        "Same-quality sparse superiority without same-split quality gates and repeated evidence.",
        "Customer-proven cost savings without external pilot results.",
    ]


def _controlled_comparison_ready(fields: dict[str, Any]) -> bool:
    return (
        _coerce_int(fields.get("run_count")) >= 2
        and _coerce_int(fields.get("baselines")) >= 1
        and _known(fields.get("dataset_name"))
        and _known(fields.get("split_id"))
        and _known(fields.get("split_seed"))
    )


def _resolve_report_dir(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _relative_ref(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        value = completed.stdout.strip()
        return value if len(value) >= 7 else "unknown"
    except Exception:
        return "unknown"


def _first_run_matching(runs: list[dict[str, Any]], text: str) -> dict[str, Any]:
    text = text.lower()
    for run in runs:
        if text in str(run.get("run_id", "")).lower():
            return run
    return {}


def _best_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    return max(
        runs,
        key=lambda run: (
            _coerce_float(run.get("gen_verifier_pass_rate")),
            _coerce_float(run.get("gen_exact_match_rate")),
            _coerce_float(run.get("tokens_per_sec")),
        ),
    )


def _is_baseline_run(run: dict[str, Any]) -> bool:
    run_id = str(run.get("run_id", "")).lower()
    return "dense" in run_id or "mop-full" in run_id or "mop_full" in run_id


def _acceptance_passed(acceptance: dict[str, Any]) -> bool | None:
    if not acceptance:
        return None
    if "overall_passed" in acceptance:
        return bool(acceptance.get("overall_passed"))
    checks = dict(acceptance.get("checks") or {})
    if checks:
        return all(bool(value) for value in checks.values())
    if "passed" in acceptance:
        return bool(acceptance.get("passed"))
    return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _coerce_float(numerator)
    bottom = _coerce_float(denominator)
    if bottom == 0.0:
        return None
    return round(top / bottom, 4)


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(_coerce_float(left) - _coerce_float(right), 6)


def _coerce_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _academic_rank(level: str) -> int:
    return ACADEMIC_LEVELS.get(level, -1)


def _product_rank(level: str) -> int:
    return PRODUCT_LEVELS.get(level, -2)


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _known(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "unknown", "none", "null"}
