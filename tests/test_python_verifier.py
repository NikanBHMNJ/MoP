"""Tests for the local Python verifier."""

from __future__ import annotations

from mopforge.verify import VerificationResult, verify_python_solution


def test_verifier_passes_correct_solution() -> None:
    result = verify_python_solution(
        "def add(a, b):\n    return a + b",
        "assert add(2, 3) == 5",
    )

    assert isinstance(result, VerificationResult)
    assert result.passed is True
    assert result.exit_code == 0
    assert result["returncode"] == 0
    assert result.error_type is None


def test_verifier_fails_incorrect_solution() -> None:
    result = verify_python_solution(
        "def add(a, b):\n    return a - b",
        "assert add(2, 3) == 5",
    )

    assert result.passed is False
    assert result.exit_code != 0
    assert result.error_type == "test_failure"


def test_verifier_reports_timeout() -> None:
    result = verify_python_solution("while True:\n    pass", "", timeout_seconds=1)

    assert result.passed is False
    assert result.timeout is True
    assert result.exit_code is None
    assert result.error_type == "timeout"
    assert result.duration_ms is not None


def test_verifier_reports_syntax_error() -> None:
    result = verify_python_solution("def broken(:\n    pass", "")

    assert result.passed is False
    assert result.error_type == "syntax_error"


def test_verifier_reports_runtime_error() -> None:
    result = verify_python_solution("def fail():\n    return missing_name", "fail()")

    assert result.passed is False
    assert result.error_type == "runtime_error"
