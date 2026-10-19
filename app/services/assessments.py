"""The user's own label for a batch of papers (CONTEXT: Assessment)."""
from __future__ import annotations

_MAX_ID = 255


class AssessmentIdError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def normalize_assessment_id(value) -> str | None:
    """A stripped assessment identifier, None when absent; invalid input raises."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AssessmentIdError("assessment_id_invalid", "Assessment identifier is invalid")
    text = value.strip()
    if not text:
        return None
    if len(text) > _MAX_ID or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise AssessmentIdError("assessment_id_invalid", "Assessment identifier is invalid")
    return text
