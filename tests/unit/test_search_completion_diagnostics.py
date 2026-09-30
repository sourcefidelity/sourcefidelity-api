"""Offline reproductions of completion limits, not live negative acceptance."""
from unittest.mock import Mock

from app.services.candidate_budget import candidate_budget_scope
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.retrieval.semantic_scholar import SemanticScholarRetriever
from app.services.search.base import SearchResult


def test_unused_global_budget_does_not_erase_tier_unattempted_candidates():
    search = WebSearchRetriever.__new__(WebSearchRetriever)
    candidates = {str(i): (SearchResult(f'https://example.org/{i}', 'Example title', ''),
                           'Example title', 'exa') for i in range(8)}
    with candidate_budget_scope(32) as budget:
        result = search._locations_result(candidates, queries_run=['Example title'], doi=None,
            title='Example title', author='Writer', year='2020', search_attempts=[])
        assert budget.snapshot()['attempted'] == 0
    assert len(result.locations) == 5
    assert len(result.metadata['candidate_dispositions']) == 3
    assert all(d['outcome'] == 'not_attempted' and d['reason_code'] == 'bounded_location_limit'
               for d in result.metadata['candidate_dispositions'])


def test_deferred_doi_no_prefetch_is_not_a_network_failure():
    adapter = SemanticScholarRetriever(health_store=Mock())
    adapter._request = Mock()
    before = adapter.search_by_doi('10.1234/example')
    assert not before.success and 'not prefetched' in before.error
    adapter._request.assert_not_called()
    adapter._request.return_value.status_code = 200
    adapter._request.return_value.json.return_value = [dict(title='Example source', year=2020,
        authors=[dict(name='Example Writer')], externalIds={'DOI':'10.1234/example'})]
    assert adapter.prefetch_dois(['10.1234/example']) == 1
    after = adapter.search_by_doi('10.1234/example')
    assert after.success and after.title == 'Example source'
    adapter._request.assert_called_once()
