"""Simple deterministic code extraction from generated text."""

from __future__ import annotations

import re


_FENCE_RE = re.compile(
    r"```(?P<label>[A-Za-z0-9_-]*)[ \t]*\r?\n(?P<code>.*?)```",
    re.DOTALL,
)
_FIXED_CODE_RE = re.compile(
    r"<fixed_code>[ \t]*\r?\n?(?P<code>.*?)\r?\n?[ \t]*</fixed_code>",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_code(text: str) -> str:
    """Extract Python code from fenced or raw generated text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    stripped = text.strip()
    if not stripped:
        return ""

    fixed_code = _FIXED_CODE_RE.search(stripped)
    if fixed_code:
        return fixed_code.group("code").strip()

    matches = list(_FENCE_RE.finditer(stripped))
    for match in matches:
        if match.group("label").lower() in {"python", "py"}:
            return match.group("code").strip()
    if matches:
        return matches[0].group("code").strip()
    return stripped
