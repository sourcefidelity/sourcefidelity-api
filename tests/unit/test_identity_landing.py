"""HTML identity inspection preserves source acquisition and uncertainty boundaries."""
import json
from unittest.mock import Mock

import httpx
import pytest

from app.services import identity_landing as landing
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalResult
from app.services.candidate_budget import candidate_budget_scope
from app.services.search.transient import finalize_transient_brave, BRAVE_TRANSIENT_POLICY


def expected():
    return ExpectedBibliographicFields(title='Competition and cultural institutions',
        authors=['River, A.'], year='2020')


def location(**kwargs):
    kwargs['metadata']={'search_provider':'exa',**kwargs.get('metadata',{})}
    return AcquisitionLocation(url='https://publisher.example/item', provider='exa', **kwargs)


def response(title='Competition and cultural institutions', author='A. River'):
    html=f'<html><head><meta name="citation_title" content="{title}"><meta name="citation_author" content="{author}"><meta name="citation_publication_date" content="2020"></head><body>Private article text</body></html>'
    return httpx.Response(200, content=html.encode(), headers={'content-type':'text/html'},
        request=httpx.Request('GET','https://publisher.example/item'))


@pytest.mark.parametrize('title,outcome',[
    ('Competition and cultural institutions','unavailable'),
    ('Botanical classification of tropical trees','identity_rejected'),
    ('Competition and cultural institutions: A history','unavailable'),
    ('경쟁도입 및 경쟁심화 연구','identity_unconfirmed'),
])
def test_observed_metadata_not_search_snippet(monkeypatch,title,outcome):
    fetch=Mock(return_value=response(title));monkeypatch.setattr(landing,'safe_request',fetch)
    attempt=landing.inspect(location(metadata={'search_title':'Invented discovery title'}),expected())
    assert attempt['outcome']==outcome
    obs=attempt.get('landing_metadata_identity') or attempt['landing_metadata_observation']
    assert obs['observed']['title']==title and len(obs['content_sha256'])==64
    assert 'Private article text' not in json.dumps(attempt)
    assert fetch.call_args.kwargs['max_bytes']==512*1024
    assert fetch.call_args.kwargs['allowed_media_types']==frozenset({'text/html','application/xhtml+xml'})


@pytest.mark.parametrize('kind',[RepresentationKind.PDF,RepresentationKind.XML,RepresentationKind.PLAIN_TEXT])
def test_files_never_fetched(monkeypatch,kind):
    fetch=Mock();monkeypatch.setattr(landing,'safe_request',fetch)
    assert landing.inspect(location(representation_kind=kind),expected())['outcome']=='not_attempted'
    fetch.assert_not_called()


def test_no_author_does_not_resolve_generic_page(monkeypatch):
    monkeypatch.setattr(landing,'safe_request',lambda *a,**k:response('Image',''))
    attempt=landing.inspect(location(),expected())
    assert attempt['outcome']=='identity_unconfirmed'
    assert 'landing_metadata_observation' not in attempt


def test_budget_and_failure_do_not_supply_negative_evidence(monkeypatch):
    fetch=Mock(side_effect=TimeoutError('private URL'));monkeypatch.setattr(landing,'safe_request',fetch)
    with candidate_budget_scope(0):
        assert landing.inspect(location(),expected())['outcome']=='not_attempted'
    fetch.assert_not_called()
    result=landing.inspect(location(),expected())
    assert result['outcome']=='transport_failure' and 'private URL' not in json.dumps(result)
    assert result['reason_code']=='identity_landing_timeout'


@pytest.mark.parametrize('error,reason',[
    (landing.UnsafeUrlError('private address'),'identity_landing_unsafe_url'),
    (landing.ResponseTooLargeError('body'),'identity_landing_response_too_large'),
    (landing.ResponseMediaTypeError('pdf'),'identity_landing_non_html_response'),
    (httpx.HTTPStatusError('private URL',request=httpx.Request('GET','https://example.org'),
        response=httpx.Response(403)),'identity_landing_http_403'),
])
def test_failure_reasons_are_typed_and_content_free(monkeypatch,error,reason):
    monkeypatch.setattr(landing,'safe_request',Mock(side_effect=error))
    attempt=landing.inspect(location(),expected())
    assert attempt['reason_code']==reason
    assert 'landing_metadata_observation' not in attempt


def test_three_page_limit_shared_with_nested_operations(monkeypatch):
    fetch=Mock(return_value=response());monkeypatch.setattr(landing,'safe_request',fetch)
    @landing.bounded_landing_inspection
    def nested():
        return landing.inspect(location(),expected())
    @landing.bounded_landing_inspection
    def run():
        return [nested() for _ in range(5)]
    values=run();assert fetch.call_count==3
    assert all(v['reason_code']=='identity_landing_limit' for v in values[3:])
    nested();assert fetch.call_count==4  # New reference, not a leaked allowance.


@pytest.mark.parametrize('title,outcome',[
    ('Botanical classification of tropical trees','identity_rejected'),
    ('경쟁도입 및 경쟁심화 연구','identity_unconfirmed'),
])
def test_transient_keeps_only_independent_observation(monkeypatch,title,outcome):
    monkeypatch.setattr(landing,'safe_request',lambda *a,**k:response(title))
    loc=location(metadata={'search_title':'SECRET SEARCH TITLE'})
    attempt=landing.inspect(loc,expected())
    result=RetrievalResult(source_name='web_search',success=True,locations=[loc],
        metadata={'search_retention_policy':BRAVE_TRANSIENT_POLICY,'location_attempts':[attempt]})
    finalize_transient_brave(result)
    assert not result.locations
    encoded=json.dumps(result.metadata)
    assert 'SECRET SEARCH TITLE' not in encoded
    assert result.metadata['location_attempts'][0]['landing_metadata_observation']['observed']['title']==title
    audit=result.metadata['transient_search_audits'][0]
    assert audit['identity_rejected' if outcome=='identity_rejected' else 'unresolved']==1


@pytest.mark.parametrize('transient',[False,True])
def test_resolver_retains_observation_as_metadata_not_source(monkeypatch,transient):
    from tests.unit.test_bibliography_identity import resolver, reference
    from app.services.reference_discovery import assess_reference_discovery_trace
    class Web:
        name='web_search'
        capabilities={'web_discovery'}
        _policy_providers={'exa':object()}
        _policy_query_cache={}
        def search_reference(self, **kwargs):
            provider='brave' if transient else 'exa'
            return RetrievalResult(source_name=self.name,success=True,
                locations=[location(metadata={'search_provider':provider})],
                metadata={'search_attempts':[{'provider':provider,'query':kwargs['title'],
                    'outcome':'results','result_count':1}],
                    **({'search_retention_policy':BRAVE_TRANSIENT_POLICY} if transient else {})})
        def search_after_failed_candidates(self, **kwargs):
            return RetrievalResult(source_name=self.name,success=False,error='No further providers')
    monkeypatch.setattr(landing,'safe_request',lambda *a,**k:response())
    output=resolver(monkeypatch,[Web()]).resolve_reference(reference(title=expected().title,year='2020'),identity_only=True)
    assert not output.full_text and not output.abstract and not output.representation
    trace=output.metadata['reference_discovery_trace']
    candidates=trace['candidates']
    assert len(candidates)==1
    assert candidates[0]['observed']['title']==expected().title
    assert candidates[0]['identity_evidence_kind']=='landing_page_observation'
    assert candidates[0]['location_provenance']=='independently_acquired_content'
    assessment=assess_reference_discovery_trace(trace)
    assert assessment.ready, assessment
    assert assessment.record.outcome=='confirmed'
