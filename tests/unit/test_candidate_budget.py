from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from app.services.candidate_budget import (
    ACTIVE_CANDIDATE_BUDGET, CandidateBudget, CandidateBudgetExceeded,
    candidate_budget_scope, require_source_candidate, reserve_metadata_candidate,
)
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalResult
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
from app.services.reference_discovery import ExpectedBibliographicFields


def test_atomic_unique_candidate_accounting():
    budget = CandidateBudget(8)
    with ThreadPoolExecutor(max_workers=12) as pool:
        answers = list(pool.map(lambda i: budget.reserve('source', str(i)), range(20)))
    assert sum(answers) == 8
    assert budget.snapshot() == {'limit':8,'attempted':8,'not_attempted':12}


def test_nested_scope_threads_and_cleanup():
    with candidate_budget_scope(1) as budget:
        require_source_candidate('same')
        with candidate_budget_scope(1) as nested:
            assert nested is budget
            require_source_candidate('same')
        with pytest.raises(ValueError):
            with candidate_budget_scope(2):
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(CandidateBudgetExceeded):
                pool.submit(copy_context().run, require_source_candidate, 'different').result()
    assert ACTIVE_CANDIDATE_BUDGET.get() is None
    assert budget.snapshot()['attempted'] == budget.snapshot()['not_attempted'] == 0


@pytest.mark.parametrize('limit',[-1,True,1.5])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        CandidateBudget(limit)


def test_unattempted_metadata_preserves_observation_without_judgment():
    result = RetrievalResult(source_name='crossref',success=True,title='A source title',authors=['Rivera'],year='2024')
    with candidate_budget_scope(0):
        original = result
        result = reserve_metadata_candidate('crossref',result)
        assert original.success and 'candidate_budget_skipped' not in (original.metadata or {})
        assert result.success is False
        token = _ACTIVE_DISCOVERY_TRACE.set({'reference_id':'fixture',
            'expected':ExpectedBibliographicFields(title='A source title',authors=['Rivera'],year='2024'),
            'required':{'academic_adapter'},'queries':[],'attempts':[],'candidates':[],
            'limitations':[], 'search_policy_version':'api-first-search-v2'})
        try:
            SourceResolver._record_discovery_attempt(category='academic_adapter',provider='crossref',result=result,required=True)
            trace,record = SourceResolver._discovery_artifacts()
        finally:
            _ACTIVE_DISCOVERY_TRACE.reset(token)
    candidate = trace['candidates'][0]
    assert candidate['observed']['title'] == 'A source title'
    assert candidate['comparisons'] == []
    assert candidate['acquisition_outcome'] == 'not_attempted'
    assert record['outcome'] == 'search_incomplete'


def test_locations_and_direct_web_report_budget_without_transport(monkeypatch):
    def forbidden(*a,**kw):
        pytest.fail('No transport permitted at zero budget')
    monkeypatch.setattr('app.services.source_resolver.safe_fetch_bytes',forbidden)
    monkeypatch.setattr('app.services.source_resolver.safe_request',forbidden)
    resolver = SourceResolver.__new__(SourceResolver)
    with candidate_budget_scope(0):
        result = RetrievalResult(source_name='web_search',success=True,locations=[
            AcquisitionLocation(url='https://fixture.example/source.pdf',provider='web_search',representation_kind=RepresentationKind.PDF)])
        assert not resolver._acquire_from_locations(result)
        attempt = result.metadata['location_attempts'][0]
        assert attempt['outcome'] == 'not_attempted'
        assert attempt['reason_code'] == 'candidate_budget_exhausted'
        web = resolver._try_web_fetch('https://fixture.example/article')
        assert web.metadata['web_fetch_diagnostic']['reason'] == 'candidate_budget_exhausted'


def test_publisher_fallback_cannot_bypass_allowance(monkeypatch):
    from app.services import publisher_urls
    monkeypatch.setattr(publisher_urls,'construct_pdf_url',lambda *a:'https://fixture.example/publisher.pdf')
    monkeypatch.setattr(publisher_urls,'safe_fetch_bytes',lambda *a,**kw:pytest.fail('No publisher fetch permitted'))
    with candidate_budget_scope(0), pytest.raises(CandidateBudgetExceeded):
        publisher_urls.try_download_publisher_pdf('10.1234/fixture')


def test_parallel_metadata_is_budgeted_in_configured_order(monkeypatch):
    from types import SimpleNamespace
    resolver = SourceResolver.__new__(SourceResolver)
    sources = [SimpleNamespace(name=name) for name in ('first','second')]
    def lookup(source,*args):
        assert ACTIVE_CANDIDATE_BUDGET.get() is not None
        return RetrievalResult(source_name=source.name,success=True,title='Title',authors=['Rivera'])
    monkeypatch.setattr(resolver,'_lookup_source',lookup)
    with candidate_budget_scope(1) as budget:
        rows = resolver._lookup_structured_sources(sources,None,'Title','Rivera','2024')
        assert rows[0][1].success
        assert not rows[1][1].success
        assert budget.snapshot() == {'limit':1,'attempted':1,'not_attempted':1}


def test_library_budget_stop_retains_unattempted_flag(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings,'DOI_RESOLVER_URL','https://library.example/resolve/')
    monkeypatch.setattr('app.services.source_resolver.safe_request',lambda *a,**kw:pytest.fail('No library fetch'))
    resolver = SourceResolver.__new__(SourceResolver)
    with candidate_budget_scope(0):
        result = resolver._try_doi_resolver('10.1234/fixture')
        assert not result.success and result.metadata['candidate_budget_skipped']
