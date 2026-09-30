from types import SimpleNamespace
from dataclasses import replace
import pytest
from app.config import settings
from app.routers import sources
from app.services.source_validator import ValidationResult
from app.services.file_safety import FileSafetyReport, SafetyVerdict


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.setattr(settings, 'SOURCE_INSPECTION_ENABLED', True)
    monkeypatch.setattr(settings, 'REPORT_AUTH_MODE', 'personal_local')
    require=[]
    args=dict(content=b'pdf', principal=SimpleNamespace(scope_type='personal_owner',require=require.append),
        safety_report=FileSafetyReport(verdict=SafetyVerdict.CLEAN,
            structural_verdict=SafetyVerdict.CLEAN,malware_verdict=SafetyVerdict.CLEAN),
        messages=['insufficient_identity_evidence'], title='Example title of sufficient length',
        author='Writer',year='2020',doi=None,isbn=None,edition_or_version=None,
        source_kind='monograph',document_kind='book',expected_page_range=None)
    inspected=[]
    monkeypatch.setattr('app.services.source_resolver.SourceResolver',
        lambda: SimpleNamespace(_inspect_candidate=lambda *a,**k: inspected.append(k)))
    result=ValidationResult(True,'high','complete','digital','Accepted',source_inspection={
        'decision_applied':True,'reconciliation_version':'single-title-typo-v1'})
    calls=[]
    def validate(*a,**k):
        calls.append(k)
        k['inspection_provider'](b'pdf', None)
        return result
    monkeypatch.setattr('app.services.source_validator.validate_retrieved_pdf',validate)
    return args,result,calls,require,inspected


def test_personal_upload_reuses_existing_validator(candidate):
    args,result,calls,required,inspected=candidate
    assert sources._inspect_uncorroborated_personal_upload(**args) is result
    assert calls[0]['inspection_reconciliation'] is True
    assert required == [sources.SOURCE_REPOSITORY_WRITE_CAPABILITY]
    assert inspected[0]['title'] == args['title']


@pytest.mark.parametrize('block',['off','institution','mode','unsafe','doi','isbn','edition','collection','author','failure'])
def test_upload_boundaries_never_reach_model(candidate,monkeypatch,block):
    args,result,calls,required,inspected=candidate
    if block=='off':monkeypatch.setattr(settings,'SOURCE_INSPECTION_ENABLED',False)
    if block=='institution':args['principal'].scope_type='institution'
    if block=='mode':monkeypatch.setattr(settings,'REPORT_AUTH_MODE','institutional')
    if block=='unsafe':args['safety_report']=FileSafetyReport(verdict=SafetyVerdict.REJECTED,
        structural_verdict=SafetyVerdict.REJECTED,malware_verdict=SafetyVerdict.CLEAN)
    if block in {'doi','isbn'}:args[block]='identifier'
    if block=='edition':args['edition_or_version']='reissue'
    if block=='collection':args['source_kind']='edited_collection'
    if block=='author':args['author']=None
    if block=='failure':args['messages'].append('pdf_identity_extraction_failed')
    assert sources._inspect_uncorroborated_personal_upload(**args) is None
    assert not calls and not inspected


@pytest.mark.parametrize('field,value',[('accept',False),('completeness','uncertain'),('identity_confidence','medium'),('source_inspection',{})])
def test_observation_alone_cannot_accept_upload(candidate,field,value):
    args,result,*_=candidate
    setattr(result,field,value)
    assert sources._inspect_uncorroborated_personal_upload(**args) is None


def test_reconciled_upload_confidence_is_not_zero():
    assert sources._upload_identity_confidence(['insufficient_identity_evidence','single_title_typo_reconciled']) == .9
