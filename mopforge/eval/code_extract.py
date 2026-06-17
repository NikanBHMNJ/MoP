"""Simple deterministic code extraction from generated text."""

from __future__ import annotations

import re


_FENCE_RE = re.compile(
    r"```(?P<label>[A-Za-z0-9_-]*)[ \t]*\r?\n(?P<code>.*?)```",
    re.DOTALL,
)


def extract_python_code(text: str) -> str:
    """Extract Python code from fenced or raw generated text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    stripped = text.strip()
    if not stripped:
        return ""

    matches = list(_FENCE_RE.finditer(stripped))
    for match in matches:
        if match.group("label").lower() in {"python", "py"}:
            return match.group("code").strip()
    if matches:
        return matches[0].group("code").strip()
    return stripped
