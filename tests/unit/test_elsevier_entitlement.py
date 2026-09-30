"""Elsevier is entitled by institutional IP range, not by the API key alone.

The same key that works on campus returns AUTHORIZATION_ERROR from home, and
proxied access is not supported. Reporting that as an invalid key sent a
reader after a credential that is fine; retrying it on every reference spent
latency on a route that cannot succeed from that network. Measured over the
development corpus it failed on every attempt.
"""
from unittest.mock import Mock

import httpx
import pytest

from pydantic import SecretStr
from app.services.retrieval.elsevier import ElsevierRetriever


def _response(status_code: int, body: dict | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://api.elsevier.com/content/article/doi/10.1016/j.test.2020.01.001")
    response = httpx.Response(status_code, request=request, json=body or {})
    return response


@pytest.fixture
def retriever():
    store = Mock()
    store.record_unavailable.return_value = 3600
    return ElsevierRetriever(health_store=store), store


class TestEntitlementFailure:
    @pytest.mark.parametrize("status", [401, 403])
    def test_an_authorization_failure_is_reported_as_entitlement(
            self, monkeypatch, retriever, status):
        adapter, _ = retriever
        monkeypatch.setattr("app.services.retrieval.elsevier.httpx.get",
                            Mock(return_value=_response(status)))
        result = adapter.search_by_doi("10.1016/j.test.2020.01.001")
        assert result.success is False
        assert "entitled" in result.error.lower()
        assert "invalid api key" not in result.error.lower()

    def test_it_opens_a_cooldown_rather_than_retrying_every_reference(
            self, monkeypatch, retriever):
        adapter, store = retriever
        monkeypatch.setattr("app.services.retrieval.elsevier.httpx.get",
                            Mock(return_value=_response(401)))
        adapter.search_by_doi("10.1016/j.test.2020.01.001")
        store.record_unavailable.assert_called_once()
        assert store.record_unavailable.call_args.kwargs["status"] == "access_restricted"

    def test_the_cooldown_is_long_enough_to_outlast_a_paper(self):
        """Entitlement does not change within a run; retrying inside one is waste."""
        adapter = ElsevierRetriever()
        assert adapter.policy.cooldown_seconds >= 3600
        assert adapter.policy.max_cooldown_seconds >= adapter.policy.cooldown_seconds


class TestStillAvailableWhereEntitled:
    def test_the_adapter_is_not_disabled_by_configuration(self):
        """On campus the key alone is entitled, so a static gate would be wrong."""
        from app.services.retrieval import get_retrieval_sources
        from app.config import settings
        if "elsevier" in settings.RETRIEVAL_SOURCES:
            assert "elsevier" in [s.name for s in get_retrieval_sources()]

    def test_an_institutional_token_is_sent_when_configured(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "ELSEVIER_INST_TOKEN", SecretStr("token-value"))
        headers = ElsevierRetriever()._headers()
        assert headers["X-ELS-Insttoken"] == "token-value"

    def test_no_token_header_when_none_is_configured(self, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "ELSEVIER_INST_TOKEN", None)
        assert "X-ELS-Insttoken" not in ElsevierRetriever()._headers()


class TestDeferralIsNotAFailure:
    """A route that was never asked did not fail.

    Elsevier has no title endpoint, so a title lookup defers to a DOI found by
    another provider. The trace recorded that deferral as `operational_failure`
    -- 110 phantom failures across the development corpus, which read as a
    broken provider and drove a proposal to remove it.
    """

    def test_title_search_marks_itself_not_applicable(self):
        result = ElsevierRetriever().search_by_title_author("Some Title", "Author")
        assert result.success is False
        assert (result.metadata or {}).get("lookup_applicable") is False

    def test_a_non_elsevier_doi_is_also_not_applicable(self):
        result = ElsevierRetriever().search_by_doi("10.2307/1110566")
        assert (result.metadata or {}).get("lookup_applicable") is False

    def test_the_resolver_already_distinguishes_a_deferral(self):
        """The reason code carries it; no new outcome value was needed.

        The resolver has always mapped `lookup_applicable: False` to
        `unavailable` with reason `route_not_applicable`. Elsevier read as
        `operational_failure` only because its title search never set that
        flag -- the fix was the missing flag, not a new vocabulary.
        """
        import inspect
        from app.services import source_resolver
        source = inspect.getsource(source_resolver.SourceResolver._record_discovery_attempt)
        assert "route_not_applicable" in source
