from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import pytest

from app.services.ref_field_extractor import extract_authorless_apa_journal
from app.services.reference_discovery import ExpectedBibliographicFields, ReferenceRouteAttempt, build_reference_discovery_candidate, derive_reference_discovery_record
from app.services.retrieval.base import RetrievalResult
from app.services.required_reference_author import AUTHOR_POLICY, required_author_omissions

RAW = '(1985). A complete article title about cinema. Journal, 14(2), 3-12. https://example.org/article'


def inputs():
    ref = extract_authorless_apa_journal(RAW)
    ref.reference_id = 'r'
    expected = ExpectedBibliographicFields(title=ref.title, year=ref.year)
    candidate = build_reference_discovery_candidate(
        attempt_id='a', provider='crossref', expected=expected,
        result=RetrievalResult(source_name='crossref', success=True, title=ref.title,
                               year=ref.year, authors=['Jane Smith'], doi='10.1234/article'))
    now = datetime.now(timezone.utc)
    attempt = ReferenceRouteAttempt(attempt_id='a', provider='crossref', route_category='academic_adapter',
        required=True, permitted=True, outcome='candidate_found', started_at=now, completed_at=now)
    record = derive_reference_discovery_record(reference_id='r', expected=expected,
        required_route_categories=['academic_adapter'], queries=[], attempts=[attempt], candidates=[candidate])
    return dict(citation_format='apa', references={'r': ref}, discoveries={'r': record.model_dump(mode='json')},
        inventory={'entries':[{'reference_id':'r','reference_text_sha256':hashlib.sha256(RAW.encode()).hexdigest(),
                               'status':'supplied','rectangles':[]}]}, policy_version=AUTHOR_POLICY)


def test_independently_verified_author_omission_with_original_binding():
    data = inputs(); before = deepcopy(data)
    finding, = required_author_omissions(**data)
    assert finding['verified_authors'] == ['Jane Smith']
    assert finding['verification_basis'] == 'crossref_registration_metadata'
    assert data == before


@pytest.mark.parametrize('control', ['disabled','mla','changed_raw','changed_title','review','author_present',
    'unconfirmed','missing_attempt','search_provider','missing_author','changed_observed_author',
    'conflicting_authors','wrong_year','wrong_record','unknown_binding','changed_expected_doi'])
def test_uncertain_or_tampered_input_cannot_flag(control):
    data=inputs(); record=data['discoveries']['r']; c=record['candidates'][0]
    if control=='disabled': data['policy_version']=None
    elif control=='mla': data['citation_format']='mla'
    elif control=='changed_raw': data['references']['r'].raw_ref='Smith, J. '+RAW
    elif control=='changed_title': data['references']['r'].title='Other title'
    elif control=='review': data['references']['r'].needs_review=True
    elif control=='author_present': data['references']['r'].author='Smith, J.'
    elif control=='unconfirmed': record['outcome']='possible_match'
    elif control=='missing_attempt': record['attempts']=[]
    elif control=='search_provider': c['provider']='web_search'
    elif control=='missing_author': c['observed']['authors']=[]
    elif control=='changed_observed_author': c['observed']['authors']=['Another Person']
    elif control=='wrong_year': c['observed']['year']='1986'
    elif control=='wrong_record': record['reference_id']='other'
    elif control=='changed_expected_doi': record['expected']['doi']='10.1234/stale'
    elif control=='unknown_binding': data['inventory']['entries'][0]['reference_text_sha256']=None
    elif control=='conflicting_authors':
        from app.services.reference_discovery import _value_hash
        other=deepcopy(c); other['candidate_id']='other'; other['observed']['authors']=['Another Person']
        for comparison in other['comparisons']:
            if comparison['field_name']=='author': comparison['observed_sha256']=_value_hash('Another Person')
        record['candidates'].append(other)
    assert required_author_omissions(**data)==[]


@pytest.mark.parametrize('raw', [RAW.replace('(1985)', '(n.d.)'), RAW.replace('3-12.', '3.'),
    RAW.replace('https://example.org/article',''), 'Smith, J. '+RAW,
    'A title without any author or date. Example Press. https://example.org/book'])
def test_incomplete_or_other_forms_remain_out_of_scope(raw):
    assert extract_authorless_apa_journal(raw) is None


def test_historical_artifacts_do_not_gain_author_findings():
    from app.services.paper_extraction import PaperExtractionArtifact
    old=PaperExtractionArtifact.model_validate({'paper_version_id':'old','citation_format':'apa'})
    assert old.required_author_policy_version is None


def test_fresh_extraction_persists_author_policy():
    from app.services.paper_extraction import PaperExtractionArtifact, extract_paper_evidence
    fresh = extract_paper_evidence('A short paper.\n\nReferences\n'+RAW,
        paper_version_id='new', use_llm_boundaries=False, use_llm_atomizer=False,
        use_llm_reference_fallback=False)
    assert fresh.required_author_policy_version == AUTHOR_POLICY
    restored = PaperExtractionArtifact.model_validate_json(fresh.model_dump_json())
    assert restored.required_author_policy_version == AUTHOR_POLICY


@pytest.mark.parametrize('mode', ['positive','wrong_title','empty','failure','disabled','unavailable'])
def test_bibliography_metadata_lookup_is_bounded_and_never_completed_negative(mode):
    from types import SimpleNamespace
    from app.services.required_reference_author import discover_author_metadata
    data = inputs(); ref = data['references']['r']; calls = []
    def search(title, author):
        calls.append((title, author))
        if mode == 'failure': raise TimeoutError()
        return RetrievalResult(source_name='crossref', success=mode != 'empty',
            title='A wholly different publication' if mode=='wrong_title' else ref.title,
            year=ref.year, authors=['Jane Smith'])
    source = SimpleNamespace(name='crossref', search_by_title_author=search)
    saved = discover_author_metadata(ref, [] if mode=='unavailable' else [source],
                                    permitted=mode!='disabled')
    assert len(calls) == (0 if mode in {'disabled','unavailable'} else 1)
    assert saved['status']=='unavailable'  # Not admitted citation evidence.
    record = saved['reference_discovery']
    assert record['outcome'] != 'unlocated_after_search'
    data['discoveries']['r'] = record
    assert bool(required_author_omissions(**data)) == (mode=='positive')
    if mode in {'empty','failure','disabled','unavailable'}:
        assert record['outcome']=='search_incomplete'


@pytest.mark.parametrize('policy', [AUTHOR_POLICY, None])
def test_unlinked_bibliography_budget_and_checkpoint_reuse(monkeypatch, policy):
    from types import SimpleNamespace
    from app.services import paper_workflow as workflow
    calls = []
    def search(title, author):
        calls.append(title)
        return RetrievalResult(source_name='crossref', success=False)
    refs = []
    for i in range(3):
        ref = extract_authorless_apa_journal(RAW)
        ref.reference_id = str(i); refs.append(ref)
    artifact = SimpleNamespace(citation_claims=[], references=refs, citation_format='apa',
                               required_author_policy_version=policy)
    job = SimpleNamespace(source_results=None, stage='extracted')
    monkeypatch.setattr(workflow, '_job', lambda *args: job)
    monkeypatch.setattr(workflow, '_extraction', lambda *args: artifact)
    monkeypatch.setattr(workflow, '_enter', lambda *args: None)
    resolver = SimpleNamespace(_retrieval_sources=[SimpleNamespace(name='crossref', search_by_title_author=search)])
    session = SimpleNamespace(commit=lambda:None)
    workflow.retrieve_paper_sources(session,None,'job',resolver=resolver)
    assert len(calls)==(2 if policy else 0)
    if policy:
        assert len(job.source_results)==3
        assert job.source_results[-1]['reference_discovery']['outcome']=='search_incomplete'
    before = len(calls)
    workflow.retrieve_paper_sources(session,None,'job',resolver=resolver)
    assert len(calls)==before
