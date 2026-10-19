from unittest.mock import Mock
from collections import Counter

from app.services.retrieval.web_search import WebSearchRetriever
from app.services.retrieval.base import RepresentationKind
from app.services.search.base import SearchProvider, SearchResult


def _result(url: str = "https://example.org/source.pdf") -> SearchResult:
    return SearchResult(url=url, title="Source", snippet="", is_pdf=True)


def _retriever(primary, escalation=()) -> WebSearchRetriever:
    retriever = WebSearchRetriever.__new__(WebSearchRetriever)
    retriever._search = primary
    retriever._escalation = list(escalation)
    retriever._query_cache = {}
    retriever._query_attempts = {}
    retriever._suspended_search_providers = set()
    retriever._provider_timeout_failures = Counter()
    retriever._health_store = Mock()
    retriever._health_store.cooldown_remaining.return_value = 0
    retriever._health_store.incident.return_value = None
    retriever._health_store.claim_recovery_probe.return_value = True
    retriever._health_store.record_success.return_value = False
    retriever._health_store.record_unavailable.return_value = 180
    retriever._health_store.record_timeout.return_value = 180
    retriever.recovered_provider_keys = set()
    retriever._recovery_notifications = set()
    retriever._on_provider_recovered = Mock()
    retriever.search_metrics = Counter()
    retriever._escalation_limits = {"tavily": 50, "exa": 25}
    return retriever


def test_exact_query_cache_prevents_repeat_provider_calls() -> None:
    primary = Mock()
    primary.search.return_value = [_result()]
    retriever = _retriever(primary)

    first = retriever._run_search('"A source" filetype:pdf')
    second = retriever._run_search('"A source" filetype:pdf')

    assert first == second
    primary.search.assert_called_once()
    assert retriever.search_metrics["query_cache_hits"] == 1


def test_escalation_stops_after_first_provider_with_results() -> None:
    primary = Mock()
    primary.search.return_value = []
    tavily = Mock()
    tavily.name = "Tavily"
    tavily.search.return_value = [_result("https://example.org/tavily.pdf")]
    exa = Mock()
    exa.name = "Exa"
    retriever = _retriever(primary, (tavily, exa))

    results = retriever._run_search('"A source" filetype:pdf')

    assert results[0].url.endswith("tavily.pdf")
    tavily.search.assert_called_once()
    exa.search.assert_not_called()
    assert retriever.search_metrics["provider_calls:tavily"] == 1
    assert retriever.search_metrics["provider_calls:exa"] == 0


def test_escalation_respects_per_process_call_limits() -> None:
    primary = Mock()
    primary.search.return_value = []
    tavily = Mock()
    tavily.name = "Tavily"
    tavily.search.return_value = []
    exa = Mock()
    exa.name = "Exa"
    exa.search.return_value = []
    retriever = _retriever(primary, (tavily, exa))
    retriever._escalation_limits = {"tavily": 1, "exa": 0}

    retriever._run_search("first unique query")
    retriever._run_search("second unique query")

    tavily.search.assert_called_once()
    exa.search.assert_not_called()
    assert retriever.search_metrics["provider_calls:tavily"] == 1
    assert retriever.search_metrics["budget_skips:tavily"] == 1
    assert retriever.search_metrics["budget_skips:exa"] == 2


def test_brave_uses_its_configured_budget_without_raising_the_limit() -> None:
    from app.services.search.brave import BraveSearch

    primary = Mock()
    primary.search.return_value = []
    brave = BraveSearch("test-placeholder")
    brave.search = Mock(return_value=[])
    retriever = _retriever(primary, (brave,))
    retriever._escalation_limits = {"brave": 1}
    retriever._run_search("first unique query")
    retriever._run_search("second unique query")
    brave.search.assert_called_once()
    assert retriever.search_metrics["provider_calls:brave"] == 1
    assert retriever.search_metrics["budget_skips:brave"] == 1


class _HealthManagedProvider(SearchProvider):
    """Stand-in for a direct, health-managed (non-API-policy) provider.

    The scraper-backed provider that occupied this role was removed; the
    mechanism it exercised is still live, so it is tested through a stub
    opted into `_HEALTH_MANAGED_DIRECT_PROVIDERS`.
    """

    @property
    def name(self) -> str:
        return "StubDirect"

    def search(self, query: str, num_results: int = 10):  # pragma: no cover
        return []


def _health_managed(monkeypatch) -> _HealthManagedProvider:
    monkeypatch.setattr(
        "app.services.retrieval.web_search._HEALTH_MANAGED_DIRECT_PROVIDERS",
        frozenset({"stubdirect"}),
    )
    return _HealthManagedProvider()


def test_direct_provider_captcha_opens_persistent_cooldown(monkeypatch) -> None:
    primary = _health_managed(monkeypatch)
    primary.search = Mock(return_value=[])
    primary.last_status = "captcha"
    retriever = _retriever(primary)

    retriever._run_search("first query")
    retriever._run_search("second query")

    primary.search.assert_called_once()
    retriever._health_store.record_unavailable.assert_called_once()
    assert retriever._health_store.record_unavailable.call_args.args[0] == (
        "stubdirect"
    )
    assert retriever._health_store.record_unavailable.call_args.kwargs[
        "status"
    ] == "captcha"
    assert retriever._query_attempts["second query"][0]["outcome"] == (
        "cooldown_skipped"
    )


def test_direct_provider_success_closes_incident_and_schedules_refresh(
    monkeypatch,
) -> None:
    primary = _health_managed(monkeypatch)
    primary.search = Mock(return_value=[_result()])
    primary.last_status = "completed"
    retriever = _retriever(primary)
    retriever._health_store.incident.return_value = {"last_status": "captcha"}
    retriever._health_store.record_success.return_value = True

    assert retriever._run_search("recovery query")

    retriever._health_store.claim_recovery_probe.assert_called_once_with(
        "stubdirect"
    )
    assert retriever.recovered_provider_keys == {"stubdirect"}
    retriever._on_provider_recovered.assert_called_once_with("stubdirect")


def test_web_discovery_merges_queries_and_returns_ranked_typed_locations() -> None:
    retriever = _retriever(None)
    retriever._run_search = Mock(
        side_effect=[
            [
                SearchResult(
                    url="https://journal.example/article",
                    title="Exact Source Title",
                    snippet="Author abstract and full text",
                )
            ],
            [
                SearchResult(
                    url="https://repository.example/source.pdf",
                    title="Exact Source Title PDF",
                    snippet="Author 2020",
                    is_pdf=True,
                ),
                SearchResult(
                    url="https://mirror.example/source.pdf",
                    title="Exact Source Title",
                    snippet="Author manuscript",
                    is_pdf=True,
                ),
            ],
        ]
    )

    result = retriever.search_by_title_author("Exact Source Title", "Author", "2020")

    assert result.success
    assert len(result.locations) == 3
    assert {location.representation_kind for location in result.locations} == {
        RepresentationKind.PDF,
        RepresentationKind.HTML,
    }
    assert result.metadata["queries_run"] == [
        '"Exact Source Title" Author 2020',
        '"Exact Source Title" Author 2020 filetype:pdf',
    ]
    assert all("deterministic_score" in location.metadata for location in result.locations)
    assert all("search_provider" in location.metadata for location in result.locations)


def test_web_discovery_bounds_long_author_lists_to_first_author() -> None:
    primary = Mock()
    primary.name = "Primary"
    primary.search.return_value = []
    retriever = _retriever(primary)
    author = (
        "Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., "
        "Macherey, W., Krikun, M., Cao, Y., Gao, Q., and Dean, J."
    )

    retriever.search_by_title_author(
        "Google neural machine translation system",
        author,
        "2016",
    )

    queries = [call.args[0] for call in primary.search.call_args_list]
    assert queries
    assert all(len(query) <= 330 for query in queries)
    assert all("Schuster" not in query for query in queries)
    assert queries[0] == '"Google neural machine translation system" Wu 2016'


def test_search_provenance_retains_failed_primary_before_escalation() -> None:
    primary = Mock()
    primary.name = "Primary"
    primary.search.return_value = []
    tavily = Mock()
    tavily.name = "Tavily"
    tavily.search.return_value = [_result("https://example.org/tavily.pdf")]
    retriever = _retriever(primary, (tavily,))

    result = retriever.search_by_doi("10.1234/example")

    attempts = result.metadata["search_attempts"]
    assert len(attempts) == 6
    assert [attempt["provider"] for attempt in attempts[:2]] == ["primary", "tavily"]
    assert [attempt["outcome"] for attempt in attempts[:2]] == ["no_results", "results"]


def test_post_validation_escalation_skips_rejected_provider_tier() -> None:
    primary = Mock()
    primary.name = "Primary"
    tavily = Mock()
    tavily.name = "Tavily"
    exa = Mock()
    exa.name = "Exa"
    exa.search.return_value = [_result("https://official.example/report.pdf")]
    retriever = _retriever(primary, (tavily, exa))

    result = retriever.search_after_failed_candidates(
        doi="10.1234/report",
        title="Official report",
        author="OECD",
        year="2021",
        tried_providers={"primary", "tavily"},
    )

    assert result.success
    assert result.locations[0].url == "https://official.example/report.pdf"
    assert result.metadata["escalation_reason"] == "prior_candidates_failed_validation"
    tavily.search.assert_not_called()
    exa.search.assert_called()


def test_operational_search_failure_is_not_collapsed_into_no_results() -> None:
    primary = Mock()
    primary.name = "Primary"
    primary.last_status = "operational_failure"
    primary.search.return_value = []
    retriever = _retriever(primary)

    result = retriever.search_by_title_author("Expected Work", "Scholar", "2024")

    assert result.success is False
    assert result.metadata["candidate_count"] == 0
    assert {
        attempt["outcome"] for attempt in result.metadata["search_attempts"]
    } == {"operational_failure"}
