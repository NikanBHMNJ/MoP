"""Custom exceptions for the Knowledge Training Store."""


class LessonValidationError(ValueError):
    """Raised when a knowledge lesson does not match the KTS schema."""


class LessonStoreError(RuntimeError):
    """Raised when a lesson store operation cannot be completed."""
