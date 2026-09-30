import copy
import json
import pytest

from app.services.source_identity_confirmer import (
    SourceIdentityEvidenceBundle, SourceIdentityEvidenceItem, ExpectedSourceIdentity,
    inspect_source_observations,
)
from app.services.web_source_metadata import extract_web_source_metadata
from app.services.ref_field_extractor import extract_fields_apa


def bundle():
    return SourceIdentityEvidenceBundle(content_sha256='a'*64, page_count=20,
        expected=ExpectedSourceIdentity(source_id='s', title='Example Work', author_or_contributors='A. Writer'),
        evidence=[SourceIdentityEvidenceItem(evidence_id='e000', role='front_page', page_index=0,
            text='Example Work\nA. Writer\nSample chapter only')])


def response():
    return {'source_id':'s', 'identity':'same_work', 'representation_role':'preview',
        'completeness':'warning_found', 'differences':[], 'observations':[
            {'field':'title','evidence_id':'e000','start':0,'end':12,'quote':'Example Work'},
            {'field':'author','evidence_id':'e000','start':13,'end':22,'quote':'A. Writer'},
            {'field':'completeness','evidence_id':'e000','start':23,'end':42,'quote':'Sample chapter only'}]}


def test_shared_inspection_does_not_promote_identity_or_completeness():
    out=inspect_source_observations(bundle(),response_provider=lambda *_:response())
    assert out['status']=='complete' and out['completeness']=='warning_found'
    assert out['identity']=='same_work' and not out['decision_applied']
    assert not out['complete_source_review']


@pytest.mark.parametrize('mutation', ['quote','offset','source','evidence','complete','edition'])
def test_invalid_or_unbound_findings_fail_closed(mutation):
    value=response()
    if mutation=='quote':value['observations'][0]['quote']='Different Work'
    if mutation=='offset':value['observations'][0]['end']=500
    if mutation=='source':value['source_id']='other'
    if mutation=='evidence':value['observations'][0]['evidence_id']='e999'
    if mutation=='complete':value['completeness']='complete'
    if mutation=='edition':value['differences']=[{'field':'edition','kind':'uncertain','evidence_ids':['e000']}]
    out=inspect_source_observations(bundle(),response_provider=lambda *_:value)
    assert out['status']=='incomplete' and out['identity']=='uncertain'


def test_model_failure_is_not_a_negative_source_finding():
    def failed(*_):raise TimeoutError()
    out=inspect_source_observations(bundle(),response_provider=failed)
    assert out['status']=='incomplete' and out['completeness']=='not_established'


@pytest.mark.parametrize('title,expected',[
    ('Silver Screen (Nov 1930-Oct 1931)','1930'),
    ('Screenland (Nov. 1934–Apr. 1935)','1934'),
    ('When Were You Born? (Warner Bros. Pressbook, 1938)','1938'),
    ('A history of cinema in 1938',None)])
def test_catalog_title_dates_are_structural_not_arbitrary(title,expected):
    html=f'<html><head><meta name="citation_title" content="{title}"></head><body></body></html>'
    result=extract_web_source_metadata(html,'https://lantern.mediahist.org/catalog/example')
    assert result['year']==expected
    other=extract_web_source_metadata(html,'https://other.example/article')
    assert other['date_method']!='catalog_title_publication_interval'


def test_title_led_pressbook_preserves_student_text():
    raw='When Were You Born? (Warner Bros.). (1938). Warner Bros. Pressbook. Archive.'
    ref=extract_fields_apa(raw)
    assert ref.raw_ref==raw and ref.title=='When Were You Born?'
    assert ref.author=='Warner Bros.' and ref.source_kind=='report'


def test_back_matter_cannot_supply_required_identity():
    b=bundle()
    b.evidence[0].role='back_page'
    out=inspect_source_observations(b,response_provider=lambda *_:response())
    assert out['status']=='incomplete' and out['identity']=='uncertain'


def test_selection_cutoff_is_not_missing_source_material():
    b=bundle()
    b.evidence[0].text='Example Work\nA. Writer\nThis text ends mid'
    b.evidence[0].selection_truncated=True
    value=response()
    value['observations'][-1]={'field':'completeness','evidence_id':'e000','quote':'This text ends mid'}
    out=inspect_source_observations(b,response_provider=lambda *_:value)
    assert out['reason_code']=='selection_cutoff_not_source_warning'


def test_pdf_hook_cannot_upgrade_deterministic_validation(monkeypatch):
    from app.services import source_validator as validator
    result=validator.ValidationResult(False,'medium','uncertain','digital','review')
    monkeypatch.setattr(validator,'_validate_retrieved_pdf_deterministic',lambda *_args,**_kwargs:result)
    actual=validator.validate_retrieved_pdf(b'pdf',inspection_provider=lambda *_:{'identity':'same_work'})
    assert not actual.accept and actual.completeness=='uncertain'
    assert actual.source_inspection['identity']=='same_work'
    result.identity_confidence='rejected'
    validator.validate_retrieved_pdf(b'pdf',inspection_provider=lambda *_:pytest.fail('must not override rejection'))


def test_remote_permission_cache_and_budget_are_checked(monkeypatch):
    from app.config import settings
    from app.services.source_resolver import SourceResolver
    from app.services import llm_service
    r=SourceResolver.__new__(SourceResolver)
    r._acquisition_capabilities=None
    monkeypatch.setattr(settings,'SOURCE_INSPECTION_ENABLED',True)
    monkeypatch.setattr(settings,'SOURCE_INSPECTION_REMOTE_ALLOWED',False)
    monkeypatch.setattr(settings,'LLM_BASE_URL','https://model.example/v1')
    monkeypatch.setattr(settings,'REPORT_AUTH_MODE','personal_local')
    monkeypatch.setattr(settings,'SOURCE_INSPECTION_MAX_CALLS',1)
    calls=[]
    def model(system,user,**kwargs):
        calls.append(user)
        return {'source_id':json.loads(user)['source_id'],'identity':'uncertain',
            'representation_role':'unknown','completeness':'not_established','observations':[],'differences':[]}
    monkeypatch.setattr(llm_service,'chat_completion_json',model)
    assert r._inspect_candidate(b'Example Work')['reason_code']=='processing_not_authorized'
    assert not calls
    monkeypatch.setattr(settings,'SOURCE_INSPECTION_REMOTE_ALLOWED',True)
    first=r._inspect_candidate(b'Example Work',title='Example Work')
    assert first['status']=='complete'
    assert r._inspect_candidate(b'Example Work',title='Example Work')==first and len(calls)==1
    assert r._inspect_candidate(b'Other Work')['reason_code']=='inspection_budget_exhausted'
    r._acquisition_capabilities={'student_url'}
    assert r._inspect_candidate(b'Example Work',title='Example Work')['reason_code']=='processing_not_authorized'


def test_unconfirmed_html_does_not_stop_existing_cascade(monkeypatch):
    import httpx
    from types import SimpleNamespace
    from unittest.mock import Mock
    from app.services.source_resolver import SourceResolver
    from app.services.retrieval import RetrievalResult
    r=SourceResolver.__new__(SourceResolver)
    r._acquisition_capabilities=None
    r._lookup_cache=None
    r._backend=None
    r._repository_session_factory=None
    r._check_local_cache=Mock(return_value=RetrievalResult(source_name='cache',success=False))
    r._delete_lookup_cache=Mock()
    source=SimpleNamespace(name='web_search',capabilities={'web_discovery'})
    r._retrieval_sources=[source]
    accepted=RetrievalResult(source_name='web_search',success=True,full_text=b'validated source',metadata={'identity_confidence':'high'})
    r._try_source=Mock(return_value=accepted)
    html='<html><head><meta name="citation_title" content="Example Work"></head><body><article>'+('Substantive text for extraction. '*30)+'</article></body></html>'
    response=httpx.Response(200,text=html,request=httpx.Request('GET','https://example.org/work'),headers={'content-type':'text/html'})
    monkeypatch.setattr('app.services.source_resolver.safe_request',lambda *_args,**_kwargs:response)
    result=r.resolve(title='Example Work',author='A. Writer',year='2020',student_url='https://example.org/work')
    assert result.full_text==accepted.full_text
    r._try_source.assert_called_once()


def test_catalog_record_is_never_returned_as_full_text(monkeypatch):
    import httpx
    from app.services.source_resolver import SourceResolver
    r=SourceResolver.__new__(SourceResolver)
    html='<html><head><meta name="citation_title" content="Silver Screen (Nov 1930-Oct 1931)"><meta name="citation_author" content="Silver Screen"></head><body>'+('Catalog description '*30)+'</body></html>'
    response=httpx.Response(200,text=html,request=httpx.Request('GET','https://lantern.mediahist.org/catalog/item'),headers={'content-type':'text/html'})
    monkeypatch.setattr('app.services.source_resolver.safe_request',lambda *_args,**_kwargs:response)
    result=r._try_web_fetch(str(response.url),'Silver Screen (Nov 1930-Oct 1931)',expected_author='Silver Screen',expected_year='1930')
    assert not result.success and result.full_text is None and result.abstract is None
    assert result.metadata['location_attempts'][0]['landing_metadata_identity']


def test_ocr_size_limit_preserves_native_inspection_without_raising_limits(monkeypatch):
    from app.config import settings
    from app.services.source_resolver import SourceResolver
    from app.services import source_identity_confirmer as confirmer, llm_service
    from app.services.ocr_derivative import OcrDerivativeError
    monkeypatch.setattr(settings,'SOURCE_INSPECTION_ENABLED',True)
    monkeypatch.setattr(settings,'LLM_BASE_URL','http://127.0.0.1:9000')
    monkeypatch.setattr(settings,'PURE_SCAN_OCR_ENABLED',True)
    calls=[]
    def build(*args,**kwargs):
        calls.append(kwargs['include_ocr'])
        if kwargs['include_ocr']:
            raise OcrDerivativeError('Identity observation source exceeds its bound.')
        return bundle()
    monkeypatch.setattr(confirmer,'build_source_identity_evidence',build)
    monkeypatch.setattr(llm_service,'chat_completion_json',lambda *_args,**_kwargs:response())
    r=SourceResolver.__new__(SourceResolver)
    out=r._inspect_candidate(b'pdf',title='Example Work',pdf=True)
    assert calls==[True,False]
    assert out['status']=='complete' and not out['decision_applied']
    assert out['extraction_limitations']==['supplementary_ocr_source_size_limit']


def test_other_ocr_failures_do_not_fall_back_to_model(monkeypatch):
    from app.config import settings
    from app.services.source_resolver import SourceResolver
    from app.services import source_identity_confirmer as confirmer, llm_service
    from app.services.ocr_derivative import OcrDerivativeError
    monkeypatch.setattr(settings,'SOURCE_INSPECTION_ENABLED',True)
    monkeypatch.setattr(settings,'LLM_BASE_URL','http://127.0.0.1:9000')
    def blocked(*a,**k):raise OcrDerivativeError('Safety unavailable')
    monkeypatch.setattr(confirmer,'build_source_identity_evidence',blocked)
    calls=[]
    monkeypatch.setattr(llm_service,'chat_completion_json',lambda *a,**k:calls.append(1))
    out=SourceResolver.__new__(SourceResolver)._inspect_candidate(b'pdf',pdf=True)
    assert out['status']=='incomplete' and not calls


def test_difference_needs_a_retained_observation_for_its_own_field():
    value=response()
    value['differences']=[{'field':'year','kind':'date_role','evidence_ids':['e000']}]
    out=inspect_source_observations(bundle(),response_provider=lambda *_:value)
    assert out['status']=='incomplete' and out['reason_code']=='unbound_difference'


def test_field_bound_difference_remains_inspectable():
    value=response()
    value['differences']=[{'field':'title','kind':'typographic','evidence_ids':['e000']}]
    out=inspect_source_observations(bundle(),response_provider=lambda *_:value)
    assert out['status']=='complete' and not out['decision_applied']
