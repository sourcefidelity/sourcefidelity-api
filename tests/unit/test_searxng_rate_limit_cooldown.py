"""One SearXNG rate limit pauses the engine briefly, not for the whole paper.

Diagnosed 2026-09-29: after a single "too many requests" answer the engine was
`cooldown_skipped` 29 times for the rest of one paper. CAPTCHA and access
denial still stop the engine for the run; health-store semantics (x4 growth
per consecutive failure, reset on success) are unchanged.
"""
from unittest.mock import Mock

from app.services.retrieval.provider_runtime import ProviderHealthStore, ProviderPolicy
from app.services.retrieval.web_search import _searx_rate_limit_policy
from app.services.search.searxng import SearXNGSearch
from tests.unit.test_web_search_cascade import _result, _retriever

KEY = "searxng:google cse"


def _rate_limited_primary():
    primary = SearXNGSearch("http://searxng.invalid")
    primary.search = Mock(return_value=[])
    primary.last_unresponsive_engines = [("google cse", "too many requests")]
    primary.last_failure_reasons = [{"engine": "google cse", "category": "rate_limited"}]
    primary.last_status = "rate_limited"
    return primary


def _single_group(monkeypatch):
    monkeypatch.setattr("app.services.retrieval.web_search.settings.SEARXNG_ENGINE_GROUPS", "google cse")


def test_rate_limit_uses_the_short_initial_cooldown_without_run_long_suspension(monkeypatch):
    _single_group(monkeypatch)
    monkeypatch.setattr("app.services.retrieval.web_search.settings.SEARXNG_RATE_LIMIT_COOLDOWN_SECONDS", 30)
    retriever = _retriever(_rate_limited_primary())

    retriever._run_search("first query")

    assert retriever._suspended_searx_groups == set()
    policy = retriever._health_store.record_unavailable.call_args.args[1]
    assert policy.cooldown_seconds == 30 and policy.max_cooldown_seconds == 3600
    assert retriever._health_store.record_unavailable.call_args.kwargs["status"] == "rate_limited"


def test_engine_is_retried_once_the_short_cooldown_has_passed(monkeypatch):
    _single_group(monkeypatch)
    primary = _rate_limited_primary()
    retriever = _retriever(primary)
    retriever._run_search("first query")
    # During the pause: skipped, but still not suspended for the run.
    retriever._health_store.cooldown_remaining.return_value = 20
    retriever._health_store.incident.return_value = {"last_status": "rate_limited"}
    retriever._run_search("second query")
    assert retriever._query_attempts["second query"][0]["outcome"] == "cooldown_skipped"
    assert retriever._suspended_searx_groups == set()
    # After it: the next query probes the engine again and can succeed.
    retriever._health_store.cooldown_remaining.return_value = 0
    primary.search = Mock(return_value=[_result()])
    primary.last_failure_reasons = []
    primary.last_status = "completed"
    assert retriever._run_search("third query")
    retriever._health_store.record_success.assert_called_with(KEY)


def test_captcha_still_suspends_for_the_run(monkeypatch):
    _single_group(monkeypatch)
    primary = _rate_limited_primary()
    primary.last_failure_reasons = [{"engine": "google cse", "category": "captcha"}]
    retriever = _retriever(primary)
    retriever._run_search("first query")
    assert retriever._suspended_searx_groups == {"google cse"}
    assert retriever._health_store.record_unavailable.call_args.args[1].cooldown_seconds == 180


def test_a_non_rate_limit_cooldown_still_suspends_for_the_run(monkeypatch):
    _single_group(monkeypatch)
    retriever = _retriever(_rate_limited_primary())
    retriever._health_store.cooldown_remaining.return_value = 60
    retriever._health_store.incident.return_value = {"last_status": "captcha"}
    retriever._run_search("query")
    assert retriever._suspended_searx_groups == {"google cse"}


def test_repeated_rate_limits_grow_exponentially_and_success_resets(tmp_path):
    store = ProviderHealthStore(str(tmp_path / "health.json"))
    policy = _searx_rate_limit_policy(ProviderPolicy(cooldown_seconds=180, max_cooldown_seconds=3600))
    assert [store.record_unavailable(KEY, policy, status="rate_limited") for _ in range(5)] == [
        30, 120, 480, 1920, 3600]
    assert store.record_success(KEY) is True
    assert store.record_unavailable(KEY, policy, status="rate_limited") == 30


def test_rate_limit_policy_never_lengthens_a_shorter_configured_cooldown(monkeypatch):
    monkeypatch.setattr("app.services.retrieval.web_search.settings.SEARXNG_RATE_LIMIT_COOLDOWN_SECONDS", 600)
    policy = _searx_rate_limit_policy(ProviderPolicy(cooldown_seconds=180, max_cooldown_seconds=3600))
    assert policy.cooldown_seconds == 180
