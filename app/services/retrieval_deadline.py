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


def remaining_seconds():
    """Seconds left in this reference's budget, or None outside one."""
    deadline = _DEADLINE.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


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


# native-script-route-v1: the search in a work's own script runs after the
# reference's ordinary routes and has its own allowance, so a long ordinary
# search cannot leave it unattempted (which would keep the reference neutral).
NATIVE_ROUTE_SECONDS = 120.0


@contextmanager
def separate_deadline_scope(seconds=NATIVE_ROUTE_SECONDS):
    """A fresh allowance for one follow-on route, replacing the current one
    for its duration and restoring it afterwards."""
    token = _DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _DEADLINE.reset(token)
