"""Filtered metadata hits are not auditable empty identity searches."""
from copy import deepcopy
import httpx
import pytest
from pydantic import SecretStr
from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.openalex import OpenAlexRetriever
from app.services.retrieval.core import CoreRetriever
from app.services.source_resolver import SourceResolver
from app.services.reference_discovery import assess_reference_discovery_trace
from app.services.reference_credibility import assess_reference_credibility
from test_reference_credibility_cumulative import fixture


def adapter_response(monkeypatch,provider,payload):
    from app.config import settings
    monkeypatch.setattr(settings,'CORE_API_KEY',SecretStr('test-only'))
    adapter=OpenAlexRetriever() if provider=='openalex' else CoreRetriever()
    response=httpx.Response(200,request=httpx.Request('GET','https://example.org'),json=payload)
    monkeypatch.setattr(adapter,'_get' if provider=='openalex' else '_request',lambda *a,**k:response)
    monkeypatch.setattr(adapter,'_parse_work' if provider=='openalex' else '_parse_output',
        lambda value:RetrievalResult(source_name=provider,success=True,title='Volcanic sediment dynamics',authors=['Stone']))
    return adapter


@pytest.mark.parametrize('provider',['openalex','core'])
def test_filtered_records_retain_count_and_typed_reason(monkeypatch,provider):
    adapter=adapter_response(monkeypatch,provider,{'results':[{'id':'one'},{'id':'two'}]})
    resolver=SourceResolver.__new__(SourceResolver)
    result=resolver._lookup_source(adapter,None,'Youth culture and cinematic violence','River')
    assert not result.success
    query=result.metadata['structured_search_attempts'][0]
    assert query['result_count']==2
    assert query['reason_code']=='metadata_candidates_filtered'
    assert 'id' not in result.metadata  # Do not pretend to retain the discarded records.


@pytest.mark.parametrize('provider',['openalex','core'])
@pytest.mark.parametrize('payload',[{}, {'results':None},{'results':{}},{'results':[None]}])
def test_malformed_metadata_is_not_empty_search(monkeypatch,provider,payload):
    adapter=adapter_response(monkeypatch,provider,payload)
    result=adapter.search_by_title_author('A work title','Writer')
    assert not result.success and result.error!='No results'
    assert (result.metadata or {}).get('identity_search_result_count')!=0


@pytest.mark.parametrize('provider',['openalex','core'])
def test_empty_list_without_qualified_total_remains_unqualified(monkeypatch,provider):
    adapter=adapter_response(monkeypatch,provider,{'results':[]})
    result=SourceResolver.__new__(SourceResolver)._lookup_source(adapter,None,'A work title','Writer')
    query=result.metadata['structured_search_attempts'][0]
    assert query['reason_code']=='metadata_empty_response_unqualified'
    assert query.get('result_count') is None


@pytest.mark.parametrize('reason',['metadata_candidates_filtered','metadata_empty_response_unqualified'])
def test_unadjudicated_metadata_receipt_blocks_completed_negative(reason):
    ref,trace=fixture();before=deepcopy(trace)
    trace['queries'][1]['reason_code']=reason
    trace['queries'][1]['result_count']=2 if reason=='metadata_candidates_filtered' else None
    assert assess_reference_discovery_trace(trace).record.outcome=='search_incomplete'
    assert not assess_reference_credibility(ref,None,trace)['findings']
    # An earlier receipt without the new observation is not re-tagged.
    assert assess_reference_discovery_trace(before).record.outcome=='unlocated_after_search'


def test_qualified_openalex_zero_is_still_zero(monkeypatch):
    adapter=adapter_response(monkeypatch,'openalex',{'meta':{'count':0},'results':[]})
    result=adapter.search_by_title_author('A work title','Writer')
    assert result.metadata['identity_search_result_count']==0


def test_filtered_route_does_not_erase_independently_confirmed_identity():
    from app.services.reference_discovery import ExpectedBibliographicFields,build_reference_discovery_candidate
    from app.services.retrieval.crossref import CrossrefRetriever
    ref,trace=fixture()
    result=CrossrefRetriever()._parse_message(dict(title=[ref.title],DOI=ref.doi,
        author=[dict(family='River',given='A.')],issued={'date-parts':[[2002]]}))
    c=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),result=result)
    trace['candidates']=[c.model_dump(mode='json')]
    trace['attempts'][0]['outcome']='candidate_found'
    trace['queries'][0]['execution_outcome']='results'
    trace['queries'][1].update(reason_code='metadata_candidates_filtered',result_count=5)
    assert assess_reference_discovery_trace(trace).record.outcome=='confirmed'
    assert not assess_reference_credibility(ref,None,trace)['findings']
