"""Shared boundary for untrusted text sent to an LLM.

This module deliberately performs bounded direct-identifier redaction rather
than claiming general PII detection. Redactions preserve string length so model
locations map back to the untouched local document. Prompt builders should
then place the masked values inside a JSON data envelope and enforce a budget
over the complete system+user request.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


class LLMInputBudgetExceeded(ValueError):
    """Raised before a remote call when its complete prompt exceeds policy."""


@dataclass(frozen=True)
class RedactedText:
    text: str
    redaction_counts: dict[str, int]

    @property
    def redaction_count(self) -> int:
        return sum(self.redaction_counts.values())


_DIRECT_IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b"),
    ),
    (
        "student_identifier",
        re.compile(
            r"(?im)\b(?:student\s?id|student\s?number|student\s?no|"
            r"matriculation\s?number|candidate\s?number)\s?[:#-]\s?"
            r"[A-Z0-9][A-Z0-9\-]{4,19}"
        ),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)(?:"
            r"\+\d{1,3}(?:[\s.-]?\(?\d{2,4}\)?){2,5}"
            r"|\(\d{2,4}\)[\s.-]?\d{3,4}[\s.-]\d{3,4}"
            r"|\d{3}([\s.-])\d{3}\1\d{4}"
            r"|\d{3}\s\d{4}\s\d{4}"
            r")(?!\w)"
        ),
    ),
    (
        "labelled_name",
        re.compile(
            r"(?im)^(?:student\s+name|name|submitted\s+by)\s?:\s?[^\n]{2,100}$"
        ),
    ),
)


def _mask(match: re.Match[str]) -> str:
    """Mask non-whitespace characters without changing offsets or line breaks."""
    return "".join(character if character.isspace() else "█" for character in match.group(0))


def redact_direct_identifiers(text: str) -> RedactedText:
    """Redact a conservative set of direct identifiers, preserving length.

    This covers emails, labelled student/candidate numbers, phone-like strings
    and names in explicit cover-sheet labels. It does not claim to detect every
    person, address, institution or indirect identifier.
    """
    redacted = text or ""
    counts: dict[str, int] = {}
    for label, pattern in _DIRECT_IDENTIFIER_PATTERNS:
        if label == "phone":
            redacted, count = _redact_high_confidence_phones(redacted, pattern)
        else:
            redacted, count = pattern.subn(_mask, redacted)
        if count:
            counts[label] = count
    if len(redacted) != len(text or ""):
        raise AssertionError("LLM redaction must preserve source offsets")
    return RedactedText(text=redacted, redaction_counts=counts)


def _redact_high_confidence_phones(
    text: str, pattern: re.Pattern[str]
) -> tuple[str, int]:
    """Mask only strongly formatted 10–15 digit phone candidates.

    Scholarly prose frequently contains years, page spans, report identifiers,
    measurements and other grouped numbers.  Formatting alone is therefore not
    sufficient: a candidate also needs a plausible international phone digit
    count.  This deliberately favors precision and does not claim exhaustive
    phone detection.
    """
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        digits = sum(character.isdigit() for character in match.group(0))
        if not 10 <= digits <= 15:
            return match.group(0)
        count += 1
        return _mask(match)

    return pattern.sub(replace, text), count


def json_data_envelope(data: dict[str, Any]) -> str:
    """Serialize untrusted values as one explicit JSON data object."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    """Conservative dependency-free estimate used for pre-call batching."""
    return math.ceil((len(system_prompt) + len(user_prompt)) / 4)


def enforce_complete_prompt_budget(
    system_prompt: str,
    user_prompt: str,
    *,
    max_input_tokens: int,
) -> int:
    """Validate the complete request rather than body text alone."""
    estimated = estimate_prompt_tokens(system_prompt, user_prompt)
    if max_input_tokens <= 0 or estimated > max_input_tokens:
        raise LLMInputBudgetExceeded(
            f"complete prompt estimate {estimated} exceeds input budget {max_input_tokens}"
        )
    return estimated
