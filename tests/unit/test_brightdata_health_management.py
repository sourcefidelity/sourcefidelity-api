"""Bright Data is suspended after a rate limit rather than called through.

It is selected as a configured provider on the direct search path, where
neither the API-first policy nor the SearXNG group machinery governs it. A
full paper on 2026-09-23 met four rate limits and kept issuing queries.
"""
from app.services.retrieval.web_search import (
    _HEALTH_MANAGED_DIRECT_PROVIDERS,
    _provider_search_outcome,
)


class _Provider:
    def __init__(self, status: str) -> None:
        self.last_status = status


def test_brightdata_is_health_managed_on_the_direct_path() -> None:
    assert "brightdata" in _HEALTH_MANAGED_DIRECT_PROVIDERS


def test_a_rate_limited_search_reaches_the_suspension_branch() -> None:
    # The branch that records a cooldown keys on exactly these outcomes.
    assert _provider_search_outcome(_Provider("rate_limited"), []) == "rate_limited"


def test_a_completed_empty_search_does_not_suspend_the_provider() -> None:
    # "The engine found nothing" is an answer, and must not be read as a fault.
    assert _provider_search_outcome(_Provider("completed"), []) == "no_results"
