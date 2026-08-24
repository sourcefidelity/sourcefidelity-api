from unittest.mock import Mock
from collections import Counter

from app.services.retrieval.web_search import WebSearchRetriever
from app.services.retrieval.base import RepresentationKind
from app.services.search.base import SearchResult
from app.services.search.searxng import SearXNGSearch


def _result(url: str = "https://example.org/source.pdf") -> SearchResult:
    return SearchResult(url=url, title="Source", snippet="", is_pdf=True)


def _retriever(primary, escalation=()) -> WebSearchRetriever:
    retriever = WebSearchRetriever.__new__(WebSearchRetriever)
    retriever._search = primary
    retriever._escalation = list(escalation)
    retriever._query_cache = {}
    retriever._query_attempts = {}
    retriever._suspended_searx_groups = set()
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


def test_blocked_searx_groups_are_suspended_for_remainder_of_run() -> None:
    primary = SearXNGSearch("http://searxng.invalid")
    primary.search = Mock(return_value=[])
    primary.last_unresponsive_engines = [("mojeek", "timeout")]
    retriever = _retriever(primary)

    retriever._run_search("first unique query")
    retriever._run_search("second unique query")

    assert primary.search.call_count == 2
    assert retriever._suspended_searx_groups == {"mojeek,qwant", "startpage"}


def test_searx_does_not_query_redundant_or_retired_engines() -> None:
    primary = SearXNGSearch("http://searxng.invalid")
    primary.search = Mock(return_value=[])
    primary.last_unresponsive_engines = []
    retriever = _retriever(primary)

    retriever._run_search("source query")

    engines = [call.kwargs["engines"] for call in primary.search.call_args_list]
    assert engines == ["mojeek,qwant", "startpage"]
    assert "arxiv" not in engines
    assert "bing" not in engines
    assert "google scholar" not in engines


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
