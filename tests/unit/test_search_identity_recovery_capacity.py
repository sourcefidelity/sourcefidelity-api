"""Capacity control must leave room to recover an incorrectly supplied DOI."""
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services.candidate_budget import candidate_budget_scope
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search.base import SearchResult


@pytest.mark.parametrize('capacity', [1, 2, 5])
def test_doi_queue_preserves_title_recovery_within_inspection_limit(monkeypatch, capacity):
    monkeypatch.setattr(settings, 'SEARCH_POLICY_VERSION', 'api-first-search-v2')
    monkeypatch.setattr(settings, 'BRAVE_SEARCH_TRANSIENT_ENABLED', True)
    monkeypatch.setattr(settings, 'SEARCH_SEARXNG_FALLBACK_ENABLED', False)
    monkeypatch.setattr(settings, 'SEARCH_ESCALATION_MAX_CALLS', 'brave:3,exa:3')
    provider = Mock(name='brave', last_status='completed', last_cost_usd=None,
                    last_reason_code=None)
    provider.name='brave'
    seen=[]
    def search(query, *, num_results):
        seen.append((query,num_results))
        kind='identifier' if query=='"10.1234/wrong"' else 'title'
        return [SearchResult(f'https://candidate.example/{kind}/{i}',
                             'Other work' if kind=='identifier' else 'Expected work title', '')
                for i in range(num_results)]
    provider.search.side_effect=search
    monkeypatch.setattr('app.services.retrieval.web_search.get_search_provider',lambda *a: provider)
    retriever=WebSearchRetriever(health_store=Mock())
    with candidate_budget_scope(capacity):
        result=retriever.search_reference(doi='10.1234/wrong', title='Expected work title',
                                           author='Writer',year='2020')
    assert len(result.locations)==capacity
    assert any('/title/' in loc.url for loc in result.locations)
    assert sum(n for _,n in seen)==capacity
    assert len(seen)==(1 if capacity==1 else 2)
    assert not any('filetype:' in q for q,_ in seen)


def test_identifier_only_query_still_uses_available_capacity(monkeypatch):
    monkeypatch.setattr(settings, 'SEARCH_POLICY_VERSION', 'api-first-search-v2')
    monkeypatch.setattr(settings, 'BRAVE_SEARCH_TRANSIENT_ENABLED', True)
    monkeypatch.setattr(settings, 'SEARCH_SEARXNG_FALLBACK_ENABLED', False)
    monkeypatch.setattr(settings, 'SEARCH_ESCALATION_MAX_CALLS', 'brave:3,exa:3')
    provider=Mock(name='brave',last_status='completed',last_cost_usd=None,last_reason_code=None)
    provider.name='brave'
    provider.search.return_value=[SearchResult(f'https://candidate.example/{i}','A work','') for i in range(5)]
    monkeypatch.setattr('app.services.retrieval.web_search.get_search_provider',lambda *a:provider)
    result=WebSearchRetriever(health_store=Mock()).search_reference(
        doi='10.1234/source',title=None,author=None,year=None)
    assert len(result.locations)==5
    provider.search.assert_called_once_with('"10.1234/source"',num_results=5)
