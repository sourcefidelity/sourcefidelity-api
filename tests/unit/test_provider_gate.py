"""Shared request gate and daily search budgets (provider-gate-v1, search-daily-budget-v1)."""

import threading
import time
from unittest.mock import Mock

import pytest
import redis

from app.config import settings
from app.log_safety import safe_exception_code
from app.services.retrieval import provider_gate
from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search import search_budget
from app.services.search.policy import API_FIRST_SEARCH_POLICY
from app.services.source_resolver import _search_execution_outcome


@pytest.fixture(autouse=True)
def gate_on(monkeypatch):
    monkeypatch.setattr(settings, "PROVIDER_GATE_ENABLED", True)
    monkeypatch.setattr(provider_gate, "_local_next", {})
    monkeypatch.setattr(provider_gate, "_local_places", {})


def test_requests_start_an_interval_apart_without_redis():
    starts = []
    for _ in range(3):
        with provider_gate.turn("test", interval=0.01):
            starts.append(time.monotonic())
    # One process alone paces at a quarter of the rate: 0.04 s apart.
    assert all(later - earlier >= 0.035 for earlier, later in zip(starts, starts[1:]))


def test_a_caller_that_cannot_wait_books_nothing():
    with provider_gate.turn("test", interval=0.5):
        pass
    with pytest.raises(provider_gate.ProviderBusy):
        with provider_gate.turn("test", interval=0.5, max_wait=0.1):
            pytest.fail("ran without its turn")
    before = time.monotonic()
    with provider_gate.turn("test", interval=0.5, max_wait=5):
        waited = time.monotonic() - before
    # The refused caller did not push the next slot back by another 2 s.
    assert waited < 2.1


def test_in_flight_places_are_shared():
    active, peak, lock = [0], [0], threading.Lock()

    def request():
        with provider_gate.turn("crossref-test", interval=0, places=3, max_wait=5):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=request) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Without Redis a process keeps a quarter of the places (at least one).
    assert peak[0] == 1


def test_a_held_back_adapter_call_is_a_skip_not_a_failure():
    error = safe_exception_code(provider_gate.ProviderBusy("held"))
    result = RetrievalResult(source_name="crossref", success=False, error=error)
    assert _search_execution_outcome(result) == "cooldown_skipped"


def test_brave_limits_are_read_from_its_headers(monkeypatch):
    stored = {}
    monkeypatch.setattr(provider_gate.shared_pacing, "store_limits", lambda provider, limits: stored.update(limits))
    headers = {"X-RateLimit-Limit": "20, 20000000", "X-RateLimit-Policy": "20;w=1, 20000000;w=2592000",
               "X-RateLimit-Remaining": "19, 19999000", "X-RateLimit-Reset": "1, 1419704"}
    learned = provider_gate.learn_limits("brave", headers)
    assert learned["per_second"] == 20
    assert (learned["window_seconds"], learned["window_limit"], learned["window_remaining"]) == (2592000, 20000000, 19999000)
    assert stored["per_second"] == 20
    monkeypatch.setattr(provider_gate.shared_pacing, "stored_limits", lambda provider: learned)
    assert provider_gate.interval_for("brave") == pytest.approx(0.05)
    assert provider_gate.learn_limits("brave", {"X-RateLimit-Limit": "1"}) is None


def test_configured_rates_apply_until_a_provider_reports_its_own(monkeypatch):
    monkeypatch.setattr(settings, "PROVIDER_REQUESTS_PER_SECOND", "brave:1,exa:8")
    monkeypatch.setattr(provider_gate.shared_pacing, "stored_limits", lambda provider: None)
    assert provider_gate.interval_for("exa") == pytest.approx(0.125)
    assert provider_gate.interval_for("eric", 1.0) == 1.0


class FakeRedis:
    """The budget script's semantics, without a server."""

    def __init__(self):
        self.values = {}

    def eval(self, script, count, *args):
        keys, limits = args[:count], args[count:count * 2]
        for index, key in enumerate(keys):
            if int(limits[index]) > 0 and self.values.get(key, 0) >= int(limits[index]):
                return index + 1
        for key in keys:
            self.values[key] = self.values.get(key, 0) + 1
        return 0

    def pipeline(self):
        fake = self

        class Pipeline:
            def __init__(self):
                self.keys = []

            def decr(self, key):
                self.keys.append(key)

            def execute(self):
                for key in self.keys:
                    fake.values[key] -= 1
        return Pipeline()


def test_daily_budgets_hold_for_the_deployment_and_each_scope(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(search_budget, "_redis", lambda: fake)
    monkeypatch.setattr(settings, "WEB_SEARCH_DAILY_MAX_CALLS", "brave:3")
    monkeypatch.setattr(settings, "WEB_SEARCH_SCOPE_DAILY_MAX_CALLS", "brave:2")
    course = ("institution_course", "course-1")
    assert search_budget.reserve("brave", course)[0] is None
    assert search_budget.reserve("brave", course)[0] is None
    assert search_budget.reserve("brave", course)[0] == "scope_daily_call_ceiling"
    assert search_budget.reserve("brave", ("institution_course", "course-2"))[0] is None
    assert search_budget.reserve("brave", ("institution_course", "course-3"))[0] == "daily_call_ceiling"
    assert search_budget.reserve("exa", course) == (None, [])     # no cap set for Exa
    # A reserved call that was not made is given back.
    monkeypatch.setattr(settings, "WEB_SEARCH_DAILY_MAX_CALLS", "brave:4")
    refused, keys = search_budget.reserve("brave", ("institution_course", "course-4"))
    assert refused is None
    search_budget.release(keys)
    assert search_budget.reserve("brave", ("institution_course", "course-4"))[0] is None
    assert not any("course" in key for key in fake.values)      # scope ids are hashed


def test_a_cap_without_redis_refuses_the_call(monkeypatch):
    broken = Mock()
    broken.eval.side_effect = redis.ConnectionError("down")
    monkeypatch.setattr(search_budget, "_redis", lambda: broken)
    monkeypatch.setattr(settings, "WEB_SEARCH_DAILY_MAX_CALLS", "brave:100")
    assert search_budget.reserve("brave", None) == ("daily_budget_unavailable", [])
    monkeypatch.setattr(settings, "WEB_SEARCH_DAILY_MAX_CALLS", "")
    assert search_budget.reserve("brave", None) == (None, [])


@pytest.fixture
def retriever(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", True)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:4,exa:4")
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None) for name in ("brave", "exa")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name))
    return WebSearchRetriever(health_store=Mock()), providers


def _attempts(search):
    result = search.search_reference(doi=None, title="Frozen source control", author="Writer", year="2020")
    return result.metadata["search_attempts"]


def test_a_search_the_gate_holds_back_is_skipped_and_not_counted(retriever, monkeypatch):
    search, providers = retriever

    def busy(*args, **kwargs):
        raise provider_gate.ProviderBusy(provider_gate.ProviderBusy.reason_code)
    monkeypatch.setattr(provider_gate, "turn", busy)
    attempts = _attempts(search)
    assert attempts and all(a["outcome"] == "budget_skipped" and a["provider_calls"] == 0 for a in attempts)
    assert {a["reason_code"] for a in attempts} == {provider_gate.ProviderBusy.reason_code}
    providers["brave"].search.assert_not_called()
    assert search.search_metrics["provider_calls:brave"] == 0
    assert not search._policy_query_cache


def test_a_refused_daily_budget_skips_the_paid_call(retriever, monkeypatch):
    search, providers = retriever
    monkeypatch.setattr(search_budget, "reserve", lambda name, scope: ("daily_call_ceiling", []))
    attempts = _attempts(search)
    assert attempts and {a["reason_code"] for a in attempts} == {"daily_call_ceiling"}
    assert all(a["outcome"] == "budget_skipped" for a in attempts)
    providers["brave"].search.assert_not_called()
    providers["exa"].search.assert_not_called()
    assert search.search_metrics["provider_calls:brave"] == 0


def test_run_ceiling_is_one_locked_step(retriever, monkeypatch):
    search, _ = retriever
    monkeypatch.setattr(settings, "WEB_SEARCH_HARD_MAX_CALLS", "brave:4,exa:4")
    granted = []

    def take():
        granted.append(search._reserve_policy_call("brave"))

    threads = [threading.Thread(target=take) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert granted.count(True) == 4
