"""A budget we ran out of is not a provider that failed.

Measured over the 11-paper corpus run on 2026-09-23: 417 of 1,221 discovery
attempts were recorded as `operational_failure`, 371 of them with reason code
`route_unclassified_failure`, and 40 of 112 references had between five and ten
adapters "fail" at once. Their median elapsed time was **0.032 seconds**, and
sampled attempts ran in about one millisecond -- no request ever left the
process. The reference's own 180-second budget had expired, and the trace
blamed the providers for it, which also made every provider reliability figure
drawn from that trace wrong.
"""
import httpx
import pytest

from app.log_safety import safe_exception_code
from app.services.retrieval_deadline import (
    ReferenceBudgetExhausted,
    deadline_scope,
    remaining,
)


def test_an_expired_budget_raises_a_nameable_exception() -> None:
    with deadline_scope(0.0):
        with pytest.raises(ReferenceBudgetExhausted) as raised:
            remaining(30.0)

    assert raised.value.reason_code == "reference_elapsed_budget_timeout"


def test_the_exception_is_still_a_timeout_for_existing_handlers() -> None:
    """Callers that already catch httpx timeouts keep working unchanged."""
    assert issubclass(ReferenceBudgetExhausted, httpx.TimeoutException)


def test_the_redacted_code_names_the_budget_rather_than_the_provider() -> None:
    """`safe_exception_code` prefers `reason_code`, so the trace says what happened."""
    code = safe_exception_code(ReferenceBudgetExhausted("reference_elapsed_budget_timeout"))

    assert code == "reference_elapsed_budget_timeout"
    # The old behaviour fell through to the class name, which said nothing
    # about whose budget ran out.
    assert code != "timeout_exception"


def test_a_live_budget_still_returns_the_smaller_of_the_two_timeouts() -> None:
    with deadline_scope(5.0):
        assert remaining(30.0) <= 5.0
        assert remaining(1.0) == pytest.approx(1.0, abs=0.01)


def test_budget_exhaustion_keeps_a_required_route_incomplete() -> None:
    """The outcome must stay in the family that forces `search_incomplete`.

    Naming the cause must not quietly promote a route we never ran into one
    that completed and found nothing.
    """
    from app.services.reference_discovery import _search_is_incomplete, ReferenceRouteAttempt

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = ReferenceRouteAttempt(
        attempt_id="attempt-budget", route_category="academic_adapter",
        provider="semantic_scholar", required=True, permitted=True,
        outcome="unavailable", reason_code="route_elapsed_budget_exhausted",
        started_at=now, completed_at=now,
    )

    assert _search_is_incomplete(
        ["academic_adapter"], [attempt], [], "api-first-search-v2") is True
