"""Privacy-safe identifiers and transport logger defaults."""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar

_TRANSIENT_DISCOVERY = ContextVar("transient_discovery_logging", default=False)
_TRANSIENT_FACTORY_INSTALLED = False


@contextmanager
def transient_discovery_logging():
    """Withhold message/exception payloads during transient source processing.

    Context-local suppression preserves unrelated concurrent task logs. Apply
    at record creation, before handlers, so DEBUG and dependency logs cannot
    retain discovery URLs or their hashes. Operational metrics remain explicit.
    """
    global _TRANSIENT_FACTORY_INSTALLED
    if not _TRANSIENT_FACTORY_INSTALLED:
        previous = logging.getLogRecordFactory()
        def factory(*args, **kwargs):
            record = previous(*args, **kwargs)
            if _TRANSIENT_DISCOVERY.get():
                record.msg = "Transient discovery operation: message details withheld"
                record.args = ()
                record.exc_info = record.exc_text = record.stack_info = None
            return record
        logging.setLogRecordFactory(factory)
        _TRANSIENT_FACTORY_INSTALLED = True
    token = _TRANSIENT_DISCOVERY.set(True)
    try:
        yield
    finally:
        _TRANSIENT_DISCOVERY.reset(token)


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


_SAFE_DETAIL = re.compile(r"[A-Za-z0-9_.\[\]=;, -]{1,200}")


def bounded_detail(value: object) -> str | None:
    """Accept a detail only if it is made of fixed internal vocabulary.

    Role names, retrieval methods, channel names, model names, field paths
    and rule names all fit this shape; student, source and network text does
    not. Anything outside it is dropped rather than truncated, because a
    truncated excerpt is still an excerpt.
    """
    return value if isinstance(value, str) and _SAFE_DETAIL.fullmatch(value) else None


def safe_exception_detail(exc: BaseException) -> str | None:
    """Describe a failure using only fixed internal vocabulary.

    A pydantic ValidationError names the model, the field path and the rule
    that failed. None of those carry student, source or network text - the
    offending *value* does, and is deliberately never read. Recording only the
    exception class left a whole paper's failure undiagnosable without an
    offline replay against its own sources.
    """
    detail = bounded_detail(getattr(exc, "detail", None))
    if detail is not None:
        return detail
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return None
    try:
        reported = errors()
    except Exception:
        return None
    parts = []
    for entry in list(reported)[:3]:
        if not isinstance(entry, dict):
            continue
        location = ".".join(str(item) for item in entry.get("loc", ()))
        rule = str(entry.get("type", "") or "")
        candidate = f"{location}[{rule}]" if location else rule
        if candidate and _SAFE_DETAIL.fullmatch(candidate):
            parts.append(candidate)
    if not parts:
        return None
    title = str(getattr(exc, "title", "") or type(exc).__name__)
    if not _SAFE_DETAIL.fullmatch(title):
        title = type(exc).__name__
    return f"{title}: {'; '.join(parts)}"[:200]


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
