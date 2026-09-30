"""Prospective synthetic cumulative checks; archive access is not a gate."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import pytest

from app.services.reference_credibility import assess_reference_credibility
from app.services.reference_discovery import ReferenceDiscoveryTrace, assess_reference_discovery_trace
from app.services.schemas import ParsedReference
from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
from test_reference_credibility import kinds


def fixture():
    ref = ParsedReference(reference_id='test-cumulative',author='River, A.',year='2002',
        title='Youth culture and cinematic violence',doi='10.1234/missing',source_kind='journal_article',
        raw_ref='River, A. (2002). Youth culture and cinematic violence. Journal of Cinema, 9(2), 87–98. https://doi.org/10.1234/missing')
    now = datetime.now(timezone.utc).isoformat()
    trace = dict(reference_id=ref.reference_id,credibility_policy_version='reference-credibility-v3',
        credibility_reference_sha256=hashlib.sha256(ref.raw_ref.encode()).hexdigest(),
        search_policy_version='api-first-search-v2',required_web_providers=['brave','exa'],
        required_route_categories=['academic_adapter','bounded_web'],
        expected=dict(title=ref.title,authors=[ref.author],year=ref.year,doi=ref.doi,source_kind=ref.source_kind),
        queries=[],attempts=[],candidates=[])
    for provider in ['crossref','openalex','brave','exa']:
        category = 'academic_adapter' if provider in {'crossref','openalex'} else 'bounded_web'
        query = f'title:{ref.title} author:{ref.author}'.casefold()
        q = dict(query_id=provider,provider=provider,route_category=category,execution_provider=provider,
            execution_outcome='no_results',result_count=0,normalized_query=query,
            query_sha256=hashlib.sha256(query.encode()).hexdigest())
        trace['queries'].append(q)
        trace['attempts'].append(dict(attempt_id=provider,provider=provider,route_category=category,
            required=True,permitted=True,query_ids=[provider],outcome='no_match',started_at=now,completed_at=now))
    q = deepcopy(trace['queries'][0]);query=f'doi:{ref.doi}'
    q.update(query_id='doi-check',normalized_query=query,query_sha256=hashlib.sha256(query.encode()).hexdigest(),
             reason_code='identifier_not_registered')
    trace['queries'].append(q);trace['attempts'][0]['query_ids'].append(q['query_id'])
    return ref,trace


def test_completed_independent_checks_allow_review_without_archive():
    ref,trace = fixture()
    before = deepcopy(trace)
    # Completed-negative discovery attachment may still be suppressed. This
    # is a separately versioned review finding, not an existence determination.
    result = assess_reference_credibility(ref,None,trace)
    assert kinds(result) == ['potentially_fabricated_reference']
    finding=result['findings'][0]
    assert finding['evidence_basis']=='cumulative_identity_nonverification'
    assert finding['finding'] == 'Potentially fabricated reference. Searches using the supplied bibliographic details did not establish a matching work.'
    assert 'These searches do not prove' not in finding['finding']
    assert trace==before


@pytest.mark.parametrize('failure',['captcha','timeout','rate_limited','operational_failure','budget_skipped','cooldown_skipped'])
def test_required_failures_do_not_count_as_negative_searches(failure):
    ref,trace=fixture()
    trace['queries'][1]['execution_outcome']=failure
    assert not assess_reference_credibility(ref,None,trace)['findings']
    assert assess_reference_discovery_trace(trace).record.outcome=='search_incomplete'


def test_optional_archive_captcha_does_not_veto_other_evidence():
    ref,trace=fixture()
    q=deepcopy(trace['queries'][0]);q.update(query_id='archive',provider='journal_archive',
        execution_provider='journal_archive',execution_outcome='captcha',result_count=None)
    a=deepcopy(trace['attempts'][0]);a.update(attempt_id='archive',provider='journal_archive',
        query_ids=['archive'],required=False,outcome='access_restricted',completed_at=None)
    trace['queries'].append(q);trace['attempts'].append(a)
    assert kinds(assess_reference_credibility(ref,None,trace))==['potentially_fabricated_reference']


@pytest.mark.parametrize('mutation',['legacy','one_index','unknown_empty','title_missing','no_doi','parse_hold','missing_web','changed_journal','missing_policy','missing_category','v2'])
def test_missing_evidence_and_old_traces_do_not_gain_flags(mutation):
    ref,trace=fixture()
    if mutation=='legacy': trace.pop('credibility_policy_version')
    if mutation=='missing_policy': trace['required_web_providers']=[]
    if mutation=='missing_category': trace['required_route_categories']=[]
    if mutation=='changed_journal': ref.raw_ref=ref.raw_ref.replace('Journal of Cinema','Journal of Society')
    if mutation=='one_index': trace['queries'][1]['normalized_query']='doi:'+ref.doi
    if mutation=='unknown_empty': trace['queries'][1]['result_count']=None
    if mutation=='v2': trace['credibility_policy_version']='reference-credibility-v2'
    if mutation=='title_missing': trace['queries'][2]['normalized_query']='doi:'+ref.doi
    if mutation=='no_doi': ref.doi=''
    if mutation=='parse_hold': ref.needs_review=True
    if mutation=='missing_web': trace['queries'][3]['execution_outcome']='unknown'
    for q in trace['queries']:q['query_sha256']=hashlib.sha256(q['normalized_query'].encode()).hexdigest()
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_qualified_correction_and_unknown_candidate_prevent_flag():
    from app.services.reference_discovery import build_reference_discovery_candidate, ExpectedBibliographicFields
    ref,trace=fixture()
    message=dict(DOI='10.1234/real',title=[ref.title],author=[dict(family='River',given='A.')],issued={'date-parts':[[2002]]})
    c=build_reference_discovery_candidate(attempt_id='openalex',provider='openalex',
        expected=ExpectedBibliographicFields(**trace['expected']),result=CrossrefRetriever()._parse_message(message))
    trace['candidates']=[c.model_dump(mode='json')]
    trace['attempts'][1]['outcome']='candidate_found';trace['queries'][1]['execution_outcome']='results'
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_actual_lookup_receipts_keep_doi_check_separate_from_title_search(monkeypatch):
    ref,trace=fixture()
    source=CrossrefRetriever()
    monkeypatch.setattr(source,'search_by_doi',lambda doi:RetrievalResult(source_name='crossref',success=False,
        error='Not found',metadata={'identifier_check':'not_registered','identity_search_result_count':0}))
    monkeypatch.setattr(source,'search_by_title_author',lambda *args:RetrievalResult(source_name='crossref',success=False,
        error='No results',metadata={'identity_search_result_count':0}))
    resolver=SourceResolver.__new__(SourceResolver)
    result=resolver._lookup_source(source,ref.doi,ref.title,ref.author,ref.year)
    from app.services.reference_discovery import ExpectedBibliographicFields
    active=dict(reference_id=ref.reference_id,expected=ExpectedBibliographicFields(**trace['expected']),
        queries=[],attempts=[],candidates=[],required=set(),limitations=[])
    token=_ACTIVE_DISCOVERY_TRACE.set(active)
    try:
        resolver._record_discovery_attempt(category='academic_adapter',provider='crossref',required=True,result=result)
        assert len(active['queries'])==2
        assert active['queries'][0].reason_code=='identifier_not_registered'
        assert active['queries'][1].reason_code is None
        assert all(q.result_count==0 for q in active['queries'])
    finally:_ACTIVE_DISCOVERY_TRACE.reset(token)


def test_new_trace_field_is_omitted_from_legacy_serialization():
    _,trace=fixture();trace.pop('credibility_policy_version')
    assert 'credibility_policy_version' not in ReferenceDiscoveryTrace.model_validate(trace).model_dump(mode='json')


@pytest.mark.parametrize('provider',['crossref','openalex'])
@pytest.mark.parametrize('valid',[True,False])
def test_empty_receipt_requires_provider_response_envelope(monkeypatch,provider,valid):
    import httpx
    from app.services.retrieval.openalex import OpenAlexRetriever
    adapter=CrossrefRetriever() if provider=='crossref' else OpenAlexRetriever()
    data=({'status':'ok','message':{'items':[]}} if provider=='crossref'
          else {'meta':{'count':0},'results':[]})
    if not valid:data.pop('status' if provider=='crossref' else 'meta')
    response=httpx.Response(200,request=httpx.Request('GET','https://example.org'),json=data)
    monkeypatch.setattr(adapter,'_get',lambda *a,**kw:response)
    result=adapter.search_by_title_author('Synthetic nonexistent title','River')
    assert ((result.metadata or {}).get('identity_search_result_count')==0) is valid


@pytest.mark.parametrize('policy',['reference-credibility-v3','reference-credibility-v4'])
def test_legacy_fabrication_flag_is_presented_as_cannot_be_verified(policy):
    """Stored reports keep the former finding; it is shown under the current label."""
    from app.services.evidence_report import (_build_report_summary, _render_reference_panel_template,
                                              _render_continuous_paper, normalize_reference_findings)
    ref,trace=fixture()
    trace['credibility_policy_version']=policy
    if policy == 'reference-credibility-v4':
        trace['bounded_review_policy_version']='bounded-reference-review-v2'
    f=assess_reference_credibility(ref,None,trace)['findings'][0]
    assert f['finding_type']=='potentially_fabricated_reference'   # the audit record is unchanged
    f['source']=dict(reference_id=ref.reference_id,raw_reference=ref.raw_ref,author=ref.author,year=ref.year,title=ref.title,doi=ref.doi)
    f['rectangles']=[dict(page_index=0,x0=40,y0=50,x1=180,y1=60)]
    summaries=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,reference_practice=[f])
    assert [item['kind'] for item in summaries['evidence']]==['unverified_reference']
    assert not summaries['academic_practice'] and not summaries['reference_formatting']
    shown=normalize_reference_findings([f])[0]
    panel=_render_reference_panel_template(shown,1)
    assert 'issue-heading evidence' not in panel and 'Potentially fabricated' not in panel
    assert '<mark class="issue-heading unverified">Cannot be verified.</mark>' in panel
    surface=dict(page_dimensions=[dict(page_index=0,width=612,height=792)],page_href_template='p-{page_index}')
    html,_=_render_continuous_paper(surface,[],[shown])
    assert 'unverified-highlight' in html and 'fill:#f28b82' in html and 'academic-highlight' not in html
