"""Schemas for repair-loop failure records."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RepairFailureRecord:
    """A generated-code failure ready to become a repair lesson."""

    lesson_id: str
    original_input: str
    expected_output: str
    generated_text: str
    candidate_code: str
    failure_type: str
    verifier_stdout: str = ""
    verifier_stderr: str = ""
    exit_code: int | None = None
    timeout: bool = False
    target_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate required fields and simple types."""

        for field_name in ("lesson_id", "original_input", "expected_output", "failure_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")

        for field_name in (
            "generated_text",
            "candidate_code",
            "verifier_stdout",
            "verifier_stderr",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"{field_name} must be a string.")

        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise ValueError("exit_code must be an integer or None.")
        if not isinstance(self.timeout, bool):
            raise ValueError("timeout must be a boolean.")
        if not isinstance(self.target_modules, list) or not all(
            isinstance(module, str) and module.strip() for module in self.target_modules
        ):
            raise ValueError("target_modules must be a list of non-empty strings.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "lesson_id": self.lesson_id,
            "original_input": self.original_input,
            "expected_output": self.expected_output,
            "generated_text": self.generated_text,
            "candidate_code": self.candidate_code,
            "failure_type": self.failure_type,
            "verifier_stdout": self.verifier_stdout,
            "verifier_stderr": self.verifier_stderr,
            "exit_code": self.exit_code,
            "timeout": self.timeout,
            "target_modules": list(self.target_modules),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairFailureRecord":
        """Create a failure record from a dictionary."""

        if not isinstance(data, dict):
            raise ValueError("RepairFailureRecord data must be a dictionary.")
        return cls(
            lesson_id=data["lesson_id"],
            original_input=data["original_input"],
            expected_output=data["expected_output"],
            generated_text=data.get("generated_text", ""),
            candidate_code=data.get("candidate_code", ""),
            failure_type=data["failure_type"],
            verifier_stdout=data.get("verifier_stdout", ""),
            verifier_stderr=data.get("verifier_stderr", ""),
            exit_code=data.get("exit_code"),
            timeout=bool(data.get("timeout", False)),
            target_modules=list(data.get("target_modules", [])),
            metadata=deepcopy(data.get("metadata", {})),
        )
