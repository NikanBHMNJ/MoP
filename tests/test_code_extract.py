"""Tests for generated-code extraction."""

from __future__ import annotations

from mopforge.eval import extract_python_code


def test_extract_python_code_from_python_fence() -> None:
    text = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"

    assert extract_python_code(text) == "def add(a, b):\n    return a + b"


def test_extract_python_code_from_generic_fence() -> None:
    text = "```\nprint('hello')\n```"

    assert extract_python_code(text) == "print('hello')"


def test_extract_python_code_from_raw_text() -> None:
    text = "  def f():\n      return 1\n  "

    assert extract_python_code(text) == "def f():\n      return 1"


def test_extract_python_code_from_empty_text() -> None:
    assert extract_python_code("  \n\t") == ""
