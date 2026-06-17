"""Repair-loop helpers for turning failures into lessons."""

from mopforge.repair.collector import (
    build_repair_lessons_from_generation_results,
    failure_record_from_generation_result,
    write_repair_lessons,
)
from mopforge.repair.lesson_builder import build_repair_lesson_from_failure
from mopforge.repair.schema import RepairFailureRecord

__all__ = [
    "RepairFailureRecord",
    "build_repair_lesson_from_failure",
    "build_repair_lessons_from_generation_results",
    "failure_record_from_generation_result",
    "write_repair_lessons",
]
