"""Deterministic coding/debugging lesson builder for Python bug-fix tasks."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from mopforge.kts import KnowledgeLesson
from mopforge.verify import verify_python_solution
from mopforge.verify.python_tests import VerificationResult


BUG_CATEGORIES: tuple[str, ...] = (
    "missing return",
    "off-by-one loop/index bug",
    "wrong comparison operator",
    "wrong accumulator initialization",
    "incorrect base case in recursion",
)

_CATEGORY_SLUGS: dict[str, str] = {
    "missing return": "missing-return",
    "off-by-one loop/index bug": "off-by-one",
    "wrong comparison operator": "wrong-comparison",
    "wrong accumulator initialization": "wrong-accumulator-init",
    "incorrect base case in recursion": "bad-recursion-base-case",
}


@dataclass(frozen=True, slots=True)
class _BugFixCase:
    bug_type: str
    function_name: str
    buggy_code: str
    fixed_code: str
    test_code: str
    test_names: list[str]
    difficulty: int
    concept: str
    common_failures: list[str]


def generate_coding_bugfix_lessons(
    *,
    count_per_category: int = 10,
    verify: bool = True,
    timeout_seconds: int = 5,
) -> list[KnowledgeLesson]:
    """Generate deterministic Python coding/debugging bug-fix lessons.

    Args:
        count_per_category: Number of lessons to generate for each supported
            bug category. The default yields 50 lessons.
        verify: If True, run each fixed solution against its tests and store
            the verification result on the lesson.
        timeout_seconds: Local subprocess timeout used during verification.
    """

    if type(count_per_category) is not int or count_per_category <= 0:
        raise ValueError("count_per_category must be a positive integer.")

    lessons: list[KnowledgeLesson] = []
    for bug_type in BUG_CATEGORIES:
        for variant_index in range(count_per_category):
            lessons.append(
                generate_bugfix_lesson(
                    bug_type,
                    variant_index,
                    verify=verify,
                    timeout_seconds=timeout_seconds,
                )
            )
    return lessons


def generate_bugfix_lesson(
    bug_type: str,
    variant_index: int,
    *,
    verify: bool = True,
    timeout_seconds: int = 5,
) -> KnowledgeLesson:
    """Generate one deterministic bug-fix lesson for ``bug_type``."""

    if bug_type not in BUG_CATEGORIES:
        valid = ", ".join(BUG_CATEGORIES)
        raise ValueError(f"Unsupported bug_type {bug_type!r}. Valid values: {valid}.")
    if type(variant_index) is not int or variant_index < 0:
        raise ValueError("variant_index must be a non-negative integer.")

    case = _build_case(bug_type, variant_index)
    verification_result: VerificationResult | None = None
    if verify:
        verification_result = verify_python_solution(
            case.fixed_code, case.test_code, timeout_seconds=timeout_seconds
        )

    status = "unverified"
    if verification_result is not None:
        status = "verified" if verification_result.passed else "failed"

    verification: dict[str, object] = {
        "type": "python_tests",
        "status": status,
        "test_names": list(case.test_names),
    }
    if verification_result is not None:
        verification.update(verification_result.to_dict())

    slug = _CATEGORY_SLUGS[bug_type]
    return KnowledgeLesson(
        id=f"coding-debugging-{slug}-{variant_index:03d}",
        domain="coding",
        skill="debugging",
        subskill=bug_type,
        difficulty=case.difficulty,
        target_modules=["coding", "debugging"],
        input=case.buggy_code,
        expected_output=case.fixed_code,
        verification=verification,
        metadata={
            "language": "python",
            "function_name": case.function_name,
            "bug_type": bug_type,
            "bug_slug": slug,
            "variant_index": variant_index,
            "test_names": list(case.test_names),
            "test_code": case.test_code,
        },
        concept=case.concept,
        common_failures=list(case.common_failures),
        training_mode="supervised_bugfix",
        source="mopforge.builders.coding_bugfix",
    )


def _build_case(bug_type: str, variant_index: int) -> _BugFixCase:
    builders = {
        "missing return": _missing_return_case,
        "off-by-one loop/index bug": _off_by_one_case,
        "wrong comparison operator": _wrong_comparison_case,
        "wrong accumulator initialization": _wrong_accumulator_case,
        "incorrect base case in recursion": _bad_recursion_base_case,
    }
    return builders[bug_type](variant_index)


def _missing_return_case(index: int) -> _BugFixCase:
    function_name = f"add_order_totals_{index:02d}"
    left = index + 2
    right = index + 5
    test_name = f"test_{function_name}_returns_sum"
    buggy_code = f"""
def {function_name}(subtotal, shipping):
    total = subtotal + shipping
"""
    fixed_code = f"""
def {function_name}(subtotal, shipping):
    total = subtotal + shipping
    return total
"""
    test_code = f"""
def {test_name}():
    assert {function_name}({left}, {right}) == {left + right}
    assert {function_name}(0, {right}) == {right}


{test_name}()
"""
    return _BugFixCase(
        bug_type="missing return",
        function_name=function_name,
        buggy_code=_clean(buggy_code),
        fixed_code=_clean(fixed_code),
        test_code=_clean(test_code),
        test_names=[test_name],
        difficulty=1 + (index % 2),
        concept="A function must explicitly return the computed value.",
        common_failures=[
            "Computing a value without returning it.",
            "Relying on Python to return the final expression automatically.",
        ],
    )


def _off_by_one_case(index: int) -> _BugFixCase:
    function_name = f"sum_all_values_{index:02d}"
    values = [index + 1, index + 2, index + 3]
    expected = sum(values)
    test_name = f"test_{function_name}_includes_final_item"
    buggy_code = f"""
def {function_name}(values):
    total = 0
    for position in range(len(values) - 1):
        total += values[position]
    return total
"""
    fixed_code = f"""
def {function_name}(values):
    total = 0
    for position in range(len(values)):
        total += values[position]
    return total
"""
    test_code = f"""
def {test_name}():
    assert {function_name}({values!r}) == {expected}
    assert {function_name}([{index + 7}]) == {index + 7}


{test_name}()
"""
    return _BugFixCase(
        bug_type="off-by-one loop/index bug",
        function_name=function_name,
        buggy_code=_clean(buggy_code),
        fixed_code=_clean(fixed_code),
        test_code=_clean(test_code),
        test_names=[test_name],
        difficulty=2 + (index % 2),
        concept="Exclusive loop bounds must still cover every intended item.",
        common_failures=[
            "Subtracting one from an already exclusive upper bound.",
            "Skipping singleton collections by using range(len(items) - 1).",
        ],
    )


def _wrong_comparison_case(index: int) -> _BugFixCase:
    function_name = f"meets_minimum_score_{index:02d}"
    minimum = 50 + index
    test_name = f"test_{function_name}_allows_equal_threshold"
    buggy_code = f"""
def {function_name}(score, minimum):
    return score > minimum
"""
    fixed_code = f"""
def {function_name}(score, minimum):
    return score >= minimum
"""
    test_code = f"""
def {test_name}():
    assert {function_name}({minimum}, {minimum}) is True
    assert {function_name}({minimum + 3}, {minimum}) is True
    assert {function_name}({minimum - 1}, {minimum}) is False


{test_name}()
"""
    return _BugFixCase(
        bug_type="wrong comparison operator",
        function_name=function_name,
        buggy_code=_clean(buggy_code),
        fixed_code=_clean(fixed_code),
        test_code=_clean(test_code),
        test_names=[test_name],
        difficulty=2,
        concept="Boundary-inclusive checks need the inclusive comparison operator.",
        common_failures=[
            "Using > when the equality boundary should pass.",
            "Testing only values above and below the threshold.",
        ],
    )


def _wrong_accumulator_case(index: int) -> _BugFixCase:
    function_name = f"multiply_values_{index:02d}"
    values = [index + 2, index + 3, 2]
    expected = values[0] * values[1] * values[2]
    test_name = f"test_{function_name}_starts_from_identity"
    buggy_code = f"""
def {function_name}(values):
    product = 0
    for value in values:
        product *= value
    return product
"""
    fixed_code = f"""
def {function_name}(values):
    product = 1
    for value in values:
        product *= value
    return product
"""
    test_code = f"""
def {test_name}():
    assert {function_name}({values!r}) == {expected}
    assert {function_name}([]) == 1


{test_name}()
"""
    return _BugFixCase(
        bug_type="wrong accumulator initialization",
        function_name=function_name,
        buggy_code=_clean(buggy_code),
        fixed_code=_clean(fixed_code),
        test_code=_clean(test_code),
        test_names=[test_name],
        difficulty=2 + (index % 2),
        concept="Accumulators should start from the identity value of the operation.",
        common_failures=[
            "Initializing a multiplication accumulator to zero.",
            "Forgetting the empty-input identity value.",
        ],
    )


def _bad_recursion_base_case(index: int) -> _BugFixCase:
    function_name = f"factorial_value_{index:02d}"
    n = 3 + (index % 4)
    expected = _factorial(n)
    test_name = f"test_{function_name}_base_case_returns_identity"
    buggy_code = f"""
def {function_name}(n):
    if n == 0:
        return 0
    return n * {function_name}(n - 1)
"""
    fixed_code = f"""
def {function_name}(n):
    if n <= 1:
        return 1
    return n * {function_name}(n - 1)
"""
    test_code = f"""
def {test_name}():
    assert {function_name}(0) == 1
    assert {function_name}(1) == 1
    assert {function_name}({n}) == {expected}


{test_name}()
"""
    return _BugFixCase(
        bug_type="incorrect base case in recursion",
        function_name=function_name,
        buggy_code=_clean(buggy_code),
        fixed_code=_clean(fixed_code),
        test_code=_clean(test_code),
        test_names=[test_name],
        difficulty=3 + (index % 2),
        concept="Recursive base cases must return the operation identity.",
        common_failures=[
            "Returning zero for factorial's base case.",
            "Testing recursive cases without testing n=0 or n=1.",
        ],
    )


def _factorial(n: int) -> int:
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


def _clean(code: str) -> str:
    return dedent(code).strip() + "\n"
