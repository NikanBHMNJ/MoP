"""Schema definition for structured Knowledge Training Store lessons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from mopforge.kts.exceptions import LessonValidationError


@dataclass(slots=True)
class KnowledgeLesson:
    """A structured, validated lesson for module-targeted model training.

    The schema is intentionally small and dependency-free for v0.1. It stores
    enough metadata for future training loops to select lessons by domain,
    skill, verification state, difficulty, and target parameter modules.
    """

    id: str
    domain: str
    skill: str
    subskill: str | None
    difficulty: int
    target_modules: list[str]
    input: str
    expected_output: str
    verification: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    concept: str | None = None
    prerequisites: list[str] = field(default_factory=list)
    common_failures: list[str] = field(default_factory=list)
    training_mode: str | None = None
    source: str | None = None
    created_at: str | None = None

    VALID_VERIFICATION_STATUSES: ClassVar[set[str]] = {
        "verified",
        "verified_target",
        "unverified",
        "failed",
        "partial",
    }

    REQUIRED_FIELDS: ClassVar[set[str]] = {
        "id",
        "domain",
        "skill",
        "subskill",
        "difficulty",
        "target_modules",
        "input",
        "expected_output",
        "verification",
    }

    def __post_init__(self) -> None:
        """Validate newly constructed lessons immediately."""

        self.validate()

    @property
    def is_verified(self) -> bool:
        """Return True when the lesson verification status is ``verified``."""

        return self.verification.get("status") == "verified"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return {
            "id": self.id,
            "domain": self.domain,
            "skill": self.skill,
            "subskill": self.subskill,
            "difficulty": self.difficulty,
            "target_modules": list(self.target_modules),
            "input": self.input,
            "expected_output": self.expected_output,
            "verification": deepcopy(self.verification),
            "metadata": deepcopy(self.metadata),
            "concept": self.concept,
            "prerequisites": list(self.prerequisites),
            "common_failures": list(self.common_failures),
            "training_mode": self.training_mode,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeLesson":
        """Build and validate a lesson from a dictionary.

        Raises:
            LessonValidationError: If required fields are missing, unknown
                fields are present, or values fail validation.
        """

        if not isinstance(data, dict):
            raise LessonValidationError("Lesson data must be a dictionary.")

        schema_fields = {field.name for field in fields(cls)}
        missing = sorted(cls.REQUIRED_FIELDS - data.keys())
        if missing:
            raise LessonValidationError(
                f"Missing required lesson field(s): {', '.join(missing)}."
            )

        unexpected = sorted(set(data.keys()) - schema_fields)
        if unexpected:
            raise LessonValidationError(
                f"Unexpected lesson field(s): {', '.join(unexpected)}."
            )

        try:
            return cls(**deepcopy(data))
        except TypeError as exc:
            raise LessonValidationError(str(exc)) from exc

    def validate(self) -> None:
        """Validate the lesson in place.

        Raises:
            LessonValidationError: If the lesson violates the schema.
        """

        for field_name in ("id", "domain", "skill", "input", "expected_output"):
            value = getattr(self, field_name)
            if not _is_non_empty_string(value):
                raise LessonValidationError(
                    f"{field_name!r} must be a non-empty string."
                )

        if self.subskill is not None and not _is_non_empty_string(self.subskill):
            raise LessonValidationError(
                "'subskill' must be None or a non-empty string."
            )

        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise LessonValidationError("'difficulty' must be an integer from 1 to 5.")

        _validate_string_list("target_modules", self.target_modules, allow_empty=False)
        _validate_mapping("verification", self.verification)
        _validate_mapping("metadata", self.metadata)
        _validate_string_list("prerequisites", self.prerequisites, allow_empty=True)
        _validate_string_list(
            "common_failures", self.common_failures, allow_empty=True
        )

        verification_type = self.verification.get("type")
        verification_status = self.verification.get("status")
        if not _is_non_empty_string(verification_type):
            raise LessonValidationError(
                "'verification' must contain a non-empty string 'type'."
            )
        if verification_status not in self.VALID_VERIFICATION_STATUSES:
            valid = ", ".join(sorted(self.VALID_VERIFICATION_STATUSES))
            raise LessonValidationError(
                "'verification.status' must be one of: " f"{valid}."
            )

        for field_name in ("concept", "training_mode", "source", "created_at"):
            value = getattr(self, field_name)
            if value is not None and not _is_non_empty_string(value):
                raise LessonValidationError(
                    f"{field_name!r} must be None or a non-empty string."
                )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_mapping(field_name: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise LessonValidationError(f"{field_name!r} must be a dictionary.")


def _validate_string_list(
    field_name: str, value: Any, *, allow_empty: bool
) -> None:
    if not isinstance(value, list):
        raise LessonValidationError(f"{field_name!r} must be a list of strings.")
    if not allow_empty and not value:
        raise LessonValidationError(f"{field_name!r} must not be empty.")
    if not all(_is_non_empty_string(item) for item in value):
        raise LessonValidationError(
            f"{field_name!r} must contain only non-empty strings."
        )
