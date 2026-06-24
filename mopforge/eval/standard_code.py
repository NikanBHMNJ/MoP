"""Standard code benchmark adapters and contamination evidence."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mopforge.eval.code_extract import extract_python_code
from mopforge.generation import generate_greedy
from mopforge.models import load_gpu_checkpoint_model
from mopforge.verify import verify_python_solution


@dataclass(slots=True)
class CodeBenchmarkTask:
    task_id: str
    prompt: str
    tests: str
    canonical_solution: str | None = None
    entry_point: str | None = None
    completion_mode: str = "append_prompt"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "prompt", "tests"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.completion_mode not in {"append_prompt", "full_code"}:
            raise ValueError("completion_mode must be append_prompt or full_code.")


def load_code_benchmark(path: str | Path, *, benchmark_format: str = "auto"):
    """Load normalized tasks from HumanEval-, MBPP-, or native-style JSONL."""

    if benchmark_format not in {"auto", "humaneval", "mbpp", "native"}:
        raise ValueError("benchmark_format must be auto, humaneval, mbpp, or native.")
    tasks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            selected = benchmark_format
            if selected == "auto":
                selected = "mbpp" if "test_list" in record else "humaneval"
                if "tests" in record:
                    selected = "native"
            try:
                tasks.append(_normalize_task(record, selected, line_number))
            except Exception as exc:
                raise ValueError(f"Invalid benchmark task at line {line_number}: {exc}") from exc
    if not tasks:
        raise ValueError("Code benchmark is empty.")
    return tasks


def evaluate_code_completion(task: CodeBenchmarkTask, completion: str) -> dict[str, Any]:
    """Syntax-check and execute one trusted completion against its tests."""

    extracted = (
        extract_python_code(completion)
        if task.completion_mode == "full_code"
        or "<fixed_code>" in completion
        or "```" in completion
        else completion
    )
    candidate = (
        task.prompt + extracted
        if task.completion_mode == "append_prompt"
        else extracted
    )
    syntax_passed = True
    try:
        ast.parse(candidate)
    except SyntaxError:
        syntax_passed = False
    tests = task.tests
    if task.entry_point and re.search(r"\bdef\s+check\s*\(", tests):
        tests = f"{tests.rstrip()}\n\ncheck({task.entry_point})\n"
    verification = verify_python_solution(candidate, tests)
    canonical = task.canonical_solution
    canonical_full = None
    if canonical is not None:
        canonical_full = (
            task.prompt + canonical
            if task.completion_mode == "append_prompt"
            else canonical
        )
    return {
        "task_id": task.task_id,
        "completion": completion,
        "candidate_code": candidate,
        "syntax_passed": syntax_passed,
        "passed": verification.passed,
        "exact_match": bool(
            canonical_full is not None
            and _normalize_text(candidate) == _normalize_text(canonical_full)
        ),
        "failure_type": None if verification.passed else verification.error_type,
        "timeout": verification.timeout,
        "duration_ms": verification.duration_ms,
        "stdout": verification.stdout,
        "stderr": verification.stderr,
    }


def run_code_benchmark(
    checkpoint_path: str | Path,
    benchmark_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    benchmark_format: str = "auto",
    max_tasks: int | None = None,
    max_new_tokens: int = 256,
    device: str = "auto",
    expert_name: str | None = None,
) -> dict[str, Any]:
    """Generate one completion per task and write a pass@1 report."""

    torch = _require_torch()
    loaded = load_gpu_checkpoint_model(checkpoint_path, config_path=config_path)
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    tasks = load_code_benchmark(benchmark_path, benchmark_format=benchmark_format)
    if max_tasks is not None:
        if type(max_tasks) is not int or max_tasks <= 0:
            raise ValueError("max_tasks must be a positive integer or None.")
        tasks = tasks[:max_tasks]
    results = []
    for task in tasks:
        active_modules = [expert_name] if expert_name else None
        completion = generate_greedy(
            model,
            tokenizer,
            task.prompt,
            max_new_tokens=max_new_tokens,
            device=resolved_device,
            active_modules=active_modules,
            use_kv_cache=True,
        )
        results.append(evaluate_code_completion(task, completion))
    total = len(results)
    passed = sum(int(result["passed"]) for result in results)
    syntax = sum(int(result["syntax_passed"]) for result in results)
    exact = sum(int(result["exact_match"]) for result in results)
    report = {
        "format": "mopforge_standard_code_eval_v1",
        "benchmark_path": str(benchmark_path),
        "benchmark_format": benchmark_format,
        "checkpoint_path": str(checkpoint_path),
        "model_architecture": loaded["architecture"].to_dict(),
        "tasks": total,
        "pass_at_1": passed / total if total else 0.0,
        "syntax_pass_rate": syntax / total if total else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
        "max_new_tokens": max_new_tokens,
        "device": resolved_device,
        "warning": "Candidate benchmark code executes locally; use only trusted data.",
        "results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def audit_code_contamination(
    benchmark_tasks: Iterable[CodeBenchmarkTask],
    training_sources: Iterable[str | Path],
    *,
    ngram_size: int = 13,
    similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    """Report exact and character n-gram overlap with source documents."""

    if type(ngram_size) is not int or ngram_size <= 0:
        raise ValueError("ngram_size must be a positive integer.")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [0, 1].")
    tasks = list(benchmark_tasks)
    fingerprints = {
        task.task_id: {
            "hash": _text_hash(task.prompt + (task.canonical_solution or "")),
            "ngrams": _ngrams(task.prompt + (task.canonical_solution or ""), ngram_size),
            "max_similarity": 0.0,
            "exact_match": False,
            "source": None,
        }
        for task in tasks
    }
    documents = 0
    source_hashes = {}
    for source in training_sources:
        path = Path(source)
        source_hashes[str(path)] = _file_hash(path)
        for document in _iter_source_documents(path):
            documents += 1
            normalized_hash = _text_hash(document)
            grams = _ngrams(document, ngram_size)
            for evidence in fingerprints.values():
                exact = normalized_hash == evidence["hash"]
                union = len(grams | evidence["ngrams"])
                similarity = len(grams & evidence["ngrams"]) / union if union else 0.0
                if exact or similarity > evidence["max_similarity"]:
                    evidence["exact_match"] = bool(exact)
                    evidence["max_similarity"] = 1.0 if exact else similarity
                    evidence["source"] = str(path)
    findings = []
    for task_id, evidence in fingerprints.items():
        findings.append(
            {
                "task_id": task_id,
                "exact_match": evidence["exact_match"],
                "max_similarity": evidence["max_similarity"],
                "suspected_contamination": bool(
                    evidence["exact_match"]
                    or evidence["max_similarity"] >= similarity_threshold
                ),
                "source": evidence["source"],
            }
        )
    suspected = sum(int(item["suspected_contamination"]) for item in findings)
    return {
        "format": "mopforge_contamination_audit_v1",
        "benchmark_tasks": len(tasks),
        "training_documents": documents,
        "ngram_size": ngram_size,
        "similarity_threshold": similarity_threshold,
        "suspected_tasks": suspected,
        "passed": suspected == 0,
        "source_sha256": source_hashes,
        "findings": findings,
    }


def _normalize_task(record, selected, line_number):
    if selected == "mbpp":
        tests = record.get("test_list") or []
        if not isinstance(tests, list):
            raise ValueError("MBPP test_list must be a list.")
        prompt = str(record.get("prompt") or record.get("text") or "")
        return CodeBenchmarkTask(
            task_id=str(record.get("task_id", line_number)),
            prompt=prompt.rstrip() + "\n",
            tests="\n".join(str(value) for value in tests),
            canonical_solution=record.get("code"),
            completion_mode="full_code",
            metadata={"source_format": "mbpp"},
        )
    if selected == "native":
        return CodeBenchmarkTask(
            task_id=str(record.get("task_id", line_number)),
            prompt=str(record["prompt"]),
            tests=str(record["tests"]),
            canonical_solution=record.get("canonical_solution"),
            entry_point=record.get("entry_point"),
            completion_mode=str(record.get("completion_mode", "append_prompt")),
            metadata=dict(record.get("metadata") or {}),
        )
    return CodeBenchmarkTask(
        task_id=str(record.get("task_id", line_number)),
        prompt=str(record["prompt"]),
        tests=str(record["test"]),
        canonical_solution=record.get("canonical_solution"),
        entry_point=record.get("entry_point"),
        completion_mode="append_prompt",
        metadata={"source_format": "humaneval"},
    )


def _iter_source_documents(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() != ".jsonl":
            for line in handle:
                if line.strip():
                    yield line
            return
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("text", "content", "prompt", "input", "expected_output", "code"):
                if isinstance(record.get(key), str):
                    yield record[key]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _text_hash(value: str) -> str:
    return hashlib.sha256(_normalize_text(value).encode("utf-8")).hexdigest()


def _ngrams(value: str, size: int) -> set[str]:
    text = _normalize_text(value)
    if len(text) <= size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for code benchmark generation.") from exc
    return torch
