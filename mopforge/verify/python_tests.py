"""Minimal local Python verification helper.

This module is not a secure sandbox. It writes code to a temporary file and
runs it with the local Python interpreter. Only run trusted code locally.
Proper sandboxing is intentionally left for a later MoP-Forge milestone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result from running candidate Python code and tests locally.

    The object exposes ``to_dict`` and basic mapping-style access for callers
    that previously consumed verifier results as dictionaries.
    """

    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timeout: bool
    error_type: str | None
    duration_ms: float | None

    @property
    def returncode(self) -> int | None:
        """Backward-compatible alias for ``exit_code``."""

        return self.exit_code

    def to_dict(self) -> dict[str, bool | float | int | str | None]:
        """Return a JSON-serializable result dictionary."""

        data = asdict(self)
        data["returncode"] = self.exit_code
        return data

    def __getitem__(self, key: str) -> Any:
        """Provide lightweight dictionary-style field access."""

        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return a field value or ``default`` if the field is absent."""

        return self.to_dict().get(key, default)

    def keys(self) -> list[str]:
        """Return result field names."""

        return list(self.to_dict().keys())

    def items(self) -> list[tuple[str, bool | float | int | str | None]]:
        """Return result items."""

        return list(self.to_dict().items())


def verify_python_solution(
    candidate_code: str, test_code: str, timeout_seconds: int = 5
) -> VerificationResult:
    """Run candidate Python code plus tests in a temporary local process.

    Warning:
        This is not a secure sandbox. Candidate and test code execute with the
        current user's local permissions. Only run trusted code locally.

    TODO:
        Add a real sandbox in a future milestone before running untrusted code.
    """

    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer.")

    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mopforge_verify_") as temp_dir:
        script_path = Path(temp_dir) / "candidate_with_tests.py"
        script_path.write_text(
            f"{candidate_code}\n\n{test_code}\n", encoding="utf-8"
        )

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return VerificationResult(
                passed=False,
                exit_code=None,
                stdout=_normalize_output(exc.stdout),
                stderr=_normalize_output(exc.stderr),
                timeout=True,
                error_type="timeout",
                duration_ms=_elapsed_ms(started_at),
            )

    passed = completed.returncode == 0
    return VerificationResult(
        passed=passed,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timeout=False,
        error_type=None if passed else _classify_failure(completed.stderr),
        duration_ms=_elapsed_ms(started_at),
    )


def _normalize_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _classify_failure(stderr: str) -> str:
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "test_failure"
    return "runtime_error"
