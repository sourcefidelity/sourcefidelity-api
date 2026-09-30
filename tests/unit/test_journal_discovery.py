"""Synthetic journal metadata; no network or source admission."""
import httpx
import pytest
from app.services.journal_discovery import journal_claim, registration_checks
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate, ReferenceDiscoveryTrace
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
from app.services.schemas import ParsedReference


def reference():
    return ParsedReference(reference_id='journal-test', author='River, A.', year='2002',
        title='Youth culture and cinematic violence', source_kind='journal_article',
        raw_ref='River, A. (2002). Youth culture and cinematic violence. Journal of Cinema, 9(2), 87–98. https://doi.org/10.1234/example')


def work(**changes):
    return dict({'type':'journal-article', 'title':['Youth culture and cinematic violence'],
        'author':[{'family':'River','given':'A'}], 'issued':{'date-parts':[[2002]]},
        'DOI':'10.1234/example', 'container-title':['Journal of Cinema'],
        'ISSN':['1234-5678'], 'volume':'9','issue':'2','page':'87-98'}, **changes)


def adapter(monkeypatch, responses):
    obj = CrossrefRetriever()
    calls = []
    def get(url, params):
        calls.append((url, params))
        response = responses[len(calls)-1]
        if isinstance(response, Exception): raise response
        return httpx.Response(200, request=httpx.Request('GET',url), json={
            'status':'ok','message':{'items':response,'total-results':200}})
    monkeypatch.setattr(obj, '_get', get)
    return obj, calls


def test_original_claim_and_ambiguous_parse_abstention():
    ref = reference()
    claim = journal_claim(ref)
    assert (claim.journal,claim.volume,claim.issue,claim.pages) == ('Journal of Cinema','9','2','87–98')
    assert len(claim.reference_sha256) == 64
    assert journal_claim(ref.model_copy(update={'needs_review':True})) is None
    assert journal_claim(ref.model_copy(update={'raw_ref':ref.raw_ref.replace('87–98','98–87')})) is None


def test_bounded_registered_search_is_not_complete_archive(monkeypatch):
    obj,calls = adapter(monkeypatch, [[{'title':'Journal of Cinema','ISSN':['1234-5678']}],
        [work()], [work(),work(**{'issue':'3','DOI':'10.1234/other'})]])
    checks = list(registration_checks(obj,journal_claim(reference()),reference().title))
    assert len(calls) == 3
    assert checks[-1][0].page_overlap_dois == ['10.1234/example']
    assert len(checks[-1][1]) == 1
    assert all(c.coverage == 'registered_records_only_not_complete_journal_archive' for c,_ in checks)
    assert checks[-1][0].total_registered_records == 200


@pytest.mark.parametrize('responses, count', [([[]],1),
    ([[{'title':'Other journal','ISSN':['1234-5678']}]],1),
    ([httpx.ConnectError('offline')],1),
    ([[{'title':'Journal of Cinema','ISSN':['1234-5678']}],[],[]],3)])
def test_unresolved_and_failure_checks_never_assert_absence(monkeypatch,responses,count):
    obj,calls = adapter(monkeypatch,responses)
    result = list(registration_checks(obj,journal_claim(reference()),reference().title))
    assert len(calls) == count
    assert not any(rows for _,rows in result)
    assert all(c.coverage.endswith('not_complete_journal_archive') for c,_ in result)


def test_topical_title_without_identity_is_not_possible_match():
    expected = ExpectedBibliographicFields(title='Youth culture and cinematic violence', authors=['River, A.'], year='2002')
    result = CrossrefRetriever()._parse_message({'title':['Cinematic violence and youth culture in modern society']})
    candidate = build_reference_discovery_candidate(attempt_id='a',provider='crossref',expected=expected,result=result)
    assert not candidate.is_credible
    assert not candidate.plausible_identity_match


def test_exact_title_positive_and_corroborated_subtitle():
    obj = CrossrefRetriever()
    expected = ExpectedBibliographicFields(title=reference().title,authors=['River, A.'],year='2002')
    for title in [reference().title,reference().title+': A retrospective']:
        c = build_reference_discovery_candidate(attempt_id='a',provider='crossref',expected=expected,
            result=obj._parse_message(work(title=[title])))
        assert c.is_credible
        assert not c.has_material_conflict


def test_resolver_enrichment_preserves_checks_and_candidates(monkeypatch):
    obj,calls = adapter(monkeypatch,[[{'title':'Journal of Cinema','ISSN':['1234-5678']}],[work()],[]])
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._retrieval_sources = [obj]
    ref = reference()
    trace = dict(reference_id=ref.reference_id,expected=ExpectedBibliographicFields(
        title=ref.title,authors=[ref.author],year=ref.year,source_kind=ref.source_kind),
        required=set(),queries=[],attempts=[],candidates=[],limitations=[])
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    try:
        resolver._enrich_journal_registration(ref)
        assert len(calls) == 3
        assert len(trace['journal_checks']) == 3
        assert len(trace['candidates']) == 1
        assert trace['candidates'][0].is_credible
        payload,_ = resolver._discovery_artifacts()
        assert len(payload['journal_checks']) == 3
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_old_trace_serialization_omits_new_receipts():
    trace = ReferenceDiscoveryTrace(reference_id='legacy',expected=ExpectedBibliographicFields())
    assert 'journal_checks' not in trace.model_dump(mode='json')


@pytest.mark.parametrize('mode', ['disabled','resolved','capacity'])
def test_no_extra_calls_when_unneeded_or_not_permitted(monkeypatch,mode):
    obj,calls = adapter(monkeypatch,[])
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._retrieval_sources = [obj]
    expected = ExpectedBibliographicFields(title=reference().title,authors=['River, A.'],year='2002')
    candidate = build_reference_discovery_candidate(attempt_id='a',provider='crossref',
        expected=expected,result=obj._parse_message(work()))
    trace = dict(candidates=[candidate] if mode=='resolved' else [],queries=[],attempts=[])
    if mode=='disabled': resolver._acquisition_capabilities = set()
    if mode=='capacity': trace['queries'] = [None]*63
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    try:
        resolver._enrich_journal_registration(reference())
        assert not calls
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_publication_year_mismatch_does_not_establish_page_occupancy(monkeypatch):
    obj,_ = adapter(monkeypatch,[[{'title':'Journal of Cinema','ISSN':['1234-5678']}],[],
        [work(issued={'date-parts':[[2003]]})]])
    checks = list(registration_checks(obj,journal_claim(reference()),reference().title))
    assert checks[-1][1]  # Still retain the observed candidate for comparison.
    assert not checks[-1][0].page_overlap_dois
