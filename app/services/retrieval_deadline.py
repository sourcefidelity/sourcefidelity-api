"""Cooperative reference-local elapsed-time budget, without background work.

Transport waits retain their own timeouts. This does not forcibly interrupt DNS,
CPU work or third-party clients; it prevents starting more work after expiry.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

import httpx

POLICY = "reference-elapsed-budget-v1"
REFERENCE_SECONDS = 180.0
_DEADLINE = ContextVar("retrieval_deadline", default=None)


def expired():
    deadline = _DEADLINE.get()
    return deadline is not None and time.monotonic() >= deadline


class ReferenceBudgetExhausted(httpx.TimeoutException):
    """This reference's budget ran out before the call was made.

    Carries `reason_code` so `safe_exception_code` names it, and so a caller
    can tell "we never contacted this provider" from "this provider failed".
    Without it the trace recorded a provider operational failure for a call
    that never left the process -- measured at roughly one millisecond of
    elapsed time, 320 times across one corpus run.
    """

    reason_code = "reference_elapsed_budget_timeout"


def remaining(timeout):
    deadline = _DEADLINE.get()
    value = timeout if deadline is None else min(timeout, deadline - time.monotonic())
    if value <= 0:
        raise ReferenceBudgetExhausted("reference_elapsed_budget_timeout")
    return value


@contextmanager
def deadline_scope(seconds=REFERENCE_SECONDS):
    deadline = time.monotonic() + seconds
    existing = _DEADLINE.get()
    token = _DEADLINE.set(min(existing, deadline) if existing is not None else deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def bounded_reference(function):
    @wraps(function)
    def bounded(*args, **kwargs):
        with deadline_scope():
            return function(*args, **kwargs)
    return bounded
