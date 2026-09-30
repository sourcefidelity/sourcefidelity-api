"""Linkless references use completed identity searches, not missing-link flags."""
from copy import deepcopy
import hashlib
import pytest
from app.services.reference_credibility import assess_reference_credibility, FINDING_TEXT
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate
from app.services.retrieval.crossref import CrossrefRetriever
from test_reference_credibility_cumulative import fixture


def linkless(kind='journal_article'):
    ref,trace=fixture()
    ref.doi='';ref.url='';ref.source_kind=kind
    ref.raw_ref=ref.raw_ref.split(' https://doi.org/')[0]
    if kind in {'monograph','edited_collection'}:
        ref.raw_ref=f'{ref.author} ({ref.year}). {ref.title}. Sample Press.'
    trace['expected'].update(doi='',source_kind=kind,edition_sensitive=kind in {'monograph','edited_collection'})
    trace['credibility_reference_sha256']=hashlib.sha256(ref.raw_ref.encode()).hexdigest()
    trace['queries']=[q for q in trace['queries'] if q['query_id']!='doi-check']
    trace['attempts'][0]['query_ids'].remove('doi-check')
    if kind in {'monograph','edited_collection'}:
        q=trace['queries'][1]
        q.update(query_id='google_books',provider='google_books',execution_provider='google_books',
            normalized_query=f'intitle:{ref.title} inauthor:{ref.author.split(",",1)[0]}'.casefold())
        q['query_sha256']=hashlib.sha256(q['normalized_query'].encode()).hexdigest()
        trace['attempts'][1].update(attempt_id='google_books',provider='google_books',query_ids=['google_books'],required=False)
    return ref,trace


@pytest.mark.parametrize('kind',['journal_article','monograph','edited_collection'])
def test_linkless_complete_checks_receive_exact_finding(kind):
    ref,trace=linkless(kind);before=deepcopy(trace)
    result=assess_reference_credibility(ref,None,trace)
    assert [f['finding'] for f in result['findings']]==[FINDING_TEXT]
    assert result['findings'][0]['search_evidence']['identifier_query_ids']==[]
    assert trace==before


@pytest.mark.parametrize('kind',['journal_article','monograph'])
@pytest.mark.parametrize('failure',['timeout','captcha','rate_limited','budget_skipped','unknown'])
def test_linkless_failed_required_searches_never_flag(kind,failure):
    ref,trace=linkless(kind);trace['queries'][2]['execution_outcome']=failure
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('failure',['timeout','captcha','rate_limited','response_invalid','no_results'])
def test_book_needs_successful_known_catalog_check(failure):
    ref,trace=linkless('monograph')
    trace['queries'][1]['execution_outcome']=failure
    trace['queries'][1]['result_count']=None
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_book_does_not_substitute_generic_article_indexes_for_catalog():
    ref,trace=linkless('monograph');q=trace['queries'][1]
    q.update(provider='openalex',execution_provider='openalex',normalized_query=trace['queries'][0]['normalized_query'],
             query_sha256=trace['queries'][0]['query_sha256'])
    trace['attempts'][1]['provider']='openalex'
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('kind',['journal_article','monograph'])
@pytest.mark.parametrize('year',['2002','1987','2012'])
def test_real_work_or_other_edition_prevents_flag_without_full_text(kind,year):
    ref,trace=linkless(kind)
    c=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),
        result=CrossrefRetriever()._parse_message(dict(title=[ref.title],author=[dict(family='River',given='A.')],
            issued={'date-parts':[[int(year)]]},DOI='10.1234/real')))
    trace['candidates']=[c.model_dump(mode='json')]
    trace['attempts'][0]['outcome']='candidate_found';trace['queries'][0].update(execution_outcome='results',result_count=1)
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('variant',['subtitle','typography'])
def test_book_title_variants_prevent_negative_flag(variant):
    ref,trace=linkless('monograph')
    title=ref.title+': A retrospective' if variant=='subtitle' else ref.title.upper()
    c=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),
        result=CrossrefRetriever()._parse_message(dict(title=[title],author=[dict(family='River',given='A.')],
            issued={'date-parts':[[2012]]},DOI='10.1234/reissue')))
    trace['candidates']=[c.model_dump(mode='json')]
    trace['attempts'][0]['outcome']='candidate_found';trace['queries'][0].update(execution_outcome='results',result_count=1)
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_unrelated_registered_work_can_be_adjudicated_without_student_doi():
    ref,trace=linkless()
    c=build_reference_discovery_candidate(attempt_id='crossref',provider='crossref',
        expected=ExpectedBibliographicFields(**trace['expected']),
        result=CrossrefRetriever()._parse_message(dict(title=['Volcanic sediment transport'],author=[dict(family='Stone')],
            issued={'date-parts':[[1998]]},DOI='10.1234/other')))
    trace['candidates']=[c.model_dump(mode='json')]
    trace['attempts'][0]['outcome']='candidate_found';trace['queries'][0].update(execution_outcome='results',result_count=1)
    assert assess_reference_credibility(ref,None,trace)['findings'][0]['finding']==FINDING_TEXT
    trace['candidates'][0]['acquisition_outcome']='not_attempted'
    assert not assess_reference_credibility(ref,None,trace)['findings']


@pytest.mark.parametrize('kind',['webpage','unknown','film','book_section'])
def test_unsupported_kinds_do_not_gain_absence_flags(kind):
    ref,trace=linkless(kind)
    assert not assess_reference_credibility(ref,None,trace)['findings']


def test_article_does_not_require_apa_volume_issue_syntax():
    ref,trace=linkless()
    ref.raw_ref=f'{ref.author} "{ref.title}." Journal of Cinema, vol. 9, {ref.year}, pp. 87-98.'
    trace['credibility_reference_sha256']=hashlib.sha256(ref.raw_ref.encode()).hexdigest()
    assert assess_reference_credibility(ref,None,trace)['findings'][0]['finding']==FINDING_TEXT


@pytest.mark.parametrize('kind,title',[
    ('journal_article','Consumer safeguards in transport. *Journal of Regional Policy*'),
    ('monograph','CulturalRepresentationsandSocialPractices'),
    ('monograph','Media and society in Britain.Routledge'),
    ('monograph','Media and society in Britain. Routledge'),
])
def test_damaged_search_title_does_not_authorize_empty_search_inference(kind,title):
    # Reproduces natural extraction shapes, not private student wording.
    ref,trace=linkless(kind)
    ref.title=title;ref.raw_ref=f'{ref.author} ({ref.year}). {title}.'
    trace['expected']['title']=title
    trace['credibility_reference_sha256']=hashlib.sha256(ref.raw_ref.encode()).hexdigest()
    for q in trace['queries']:
        q['normalized_query']=(f'intitle:{title} inauthor:{ref.author.split(",",1)[0]}'
            if q['provider']=='google_books' else f'title:{title} author:{ref.author}').casefold()
        q['query_sha256']=hashlib.sha256(q['normalized_query'].encode()).hexdigest()
    before=deepcopy(trace)
    result=assess_reference_credibility(ref,None,trace)
    assert not result['findings']
    assert result['cumulative_reason_code']=='search_title_boundary_unresolved'
    assert trace==before


@pytest.mark.parametrize('title',[
    'Social media and eBay', 'U.S. media and society',
    'Media in Britain: A cultural history', 'The Sage of the Mountains',
])
def test_search_title_guard_does_not_reject_ordinary_title_punctuation(title):
    from app.services.reference_credibility import _search_title_boundary_unresolved
    ref,_=linkless('monograph');ref.title=title
    assert not _search_title_boundary_unresolved(ref)


@pytest.mark.parametrize('payload,expected',[({'totalItems':0},0),({'items':[],'totalItems':False},None),
    ({'items':[],'totalItems':10},None),({'items':[]},None)])
def test_real_catalog_adapter_receipt_counts_only_valid_empty(monkeypatch,payload,expected):
    import httpx
    from app.services.source_resolver import SourceResolver,_ACTIVE_DISCOVERY_TRACE
    ref,trace=linkless('monograph')
    monkeypatch.setattr('app.services.retrieval.google_books.httpx.get',lambda *a,**kw:
        httpx.Response(200,request=httpx.Request('GET','https://www.googleapis.com/books/v1/volumes'),json=payload))
    resolver=SourceResolver.__new__(SourceResolver)
    active=dict(reference_id=ref.reference_id,expected=ExpectedBibliographicFields(**trace['expected']),
        queries=[],attempts=[],candidates=[],required=set(),limitations=[])
    token=_ACTIVE_DISCOVERY_TRACE.set(active)
    try:
        resolver._enrich_book_editions()
        assert active['queries'][0].result_count==expected
        assert active['queries'][0].execution_outcome==('no_results' if expected==0 else 'response_invalid')
    finally:_ACTIVE_DISCOVERY_TRACE.reset(token)
