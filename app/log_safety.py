"""Privacy-safe identifiers and transport logger defaults."""

from __future__ import annotations

import hashlib
import logging
import re


_SENSITIVE_TRANSPORT_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3.connectionpool",
    "openai._base_client",
)


def private_value_id(kind: str, value: str) -> str:
    """Return a stable log identifier without retaining the original value."""
    normalized_kind = "".join(
        character
        for character in kind.strip().casefold()
        if character.isalnum() or character == "_"
    ) or "value"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{normalized_kind}_sha256={digest}"


def safe_exception_code(exc: BaseException) -> str:
    """Classify an exception without retaining URLs, queries, or document text."""
    reason_code = getattr(exc, "reason_code", None)
    if isinstance(reason_code, str) and re.fullmatch(r"[a-z0-9_]{1,80}", reason_code):
        return reason_code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return f"http_{status_code}"
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return f"http_{status_code}"
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).casefold()
    return name or "operation_failed"


def configure_sensitive_transport_logging() -> None:
    """Prevent dependency INFO logs from exposing query strings or credentials."""
    for logger_name in _SENSITIVE_TRANSPORT_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
